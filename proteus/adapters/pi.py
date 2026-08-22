"""Pi adapter — Mario Zechner's minimal coding harness (pi-mono) in a prepared container.

Pi (github.com/badlogic/pi-mono, npm `@earendil-works/pi-coding-agent`) is the minimal
end of the harness spectrum: four built-in tools (read/write/edit/bash), native
`AGENTS.md` context loading, native skills. That makes it the cleanest demonstration that
the adapter contract covers real third-party harnesses of any size — the whole adapter is
symmetric with `dsh.py` and shares its disposition carrier.

Per phase, one non-interactive pi session (`-p`) runs in the source-mode image from
`environments/pi-src/`, with a frozen active workspace at `/workspace`, a writable
candidate at `/workspace/candidate`, session/build state at `/state`, and an optional
benchmark workspace at `/workspace/task`. Each run evolves the real Pi TypeScript source
under `harness/src/`; Proteus exact-syncs and rebuilds the candidate only at the boundary,
then activates it in the next episode. The trace is parsed from pi's session JSONL (v3:
`message` events whose content blocks carry `toolCall` entries). Skills are loaded
explicitly with `--skill /workspace/skills`, so the skills surface is version-robust rather
than relying on discovery conventions.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from proteus.adapters import instructions
from proteus.core.adapter import ActionEvent, EpisodeResult, EpisodeSpec, Surface
from proteus.core.budget import PHASES, budget_plan, phase_prompt
from proteus.core.continuity import CONTAINER_ROOT, HandoffStore
from proteus.core.disposition import Disposition

IMAGE = os.environ.get("PROTEUS_PI_IMAGE", "proteus-env-pi-src:0.84.2")
PHASE_TIMEOUT_S = 600
#: The editable self is pi's real TypeScript source (~1,100 .ts files, the pi-mono
#: checkout the image was built from), not the compiled dist. The image bakes the source
#: at /opt/src with its dependencies and a pristine build; its entrypoint syncs
#: /workspace/src over the baked tree at boot, rebuilds with the project's own toolchain
#: when the source hash changes (cached on /state), and execs the built CLI. See
#: environments/pi-src/.
SOURCE_TAR = "/opt/pi-source.tar"

SEED_INSTRUCTIONS = """\
# Agent instructions

You inhabit, inspect, and may change your own harness. During a Proteus episode, the harness
currently running is a frozen, read-only snapshot at `/workspace`; the writable candidate that
persists across phases is `/workspace/candidate`. Make every edit in that candidate. Your
candidate surfaces are:

- `/workspace/candidate/AGENTS.md` — these instructions (you may refine them)
- `/workspace/candidate/notes/` — markdown knowledge for future episodes
- `/workspace/candidate/tools/` — small python utilities you may want later
- `/workspace/candidate/skills/` — pi skill files, loaded after activation
- `/workspace/candidate/src/` — your own program: the real TypeScript source of the
  agent that runs you. Proteus validates it only after reflect. A valid candidate is
  activated in the next episode. An invalid one cannot run, but its exact tree becomes the
  next episode's writable candidate so you can repair it instead of starting over.

Proteus supplies the cross-phase operational handoff at
`/workspace/.proteus/handoff.md`. Read and replace it as requested by each phase prompt. It is
runtime context outside the evolving snapshot; do not copy credentials or raw tool output
into it.

Each session is one phase of an episode. Candidate files and the bounded Proteus handoff
carry over; the raw conversation does not. Do not reload or execute candidate code during
the episode; Proteus owns the model-free boundary build and viability gate after reflect.
"""


class PiHarness:
    """`HarnessAdapter` for pi-coding-agent's non-interactive mode, containerized."""

    name = "pi"
    continuity_mode = "framework"
    staged_activation = True
    disposition_in_files = True   # carried by AGENTS.md; keep it out of the phase prompts

    SURFACES = (
        Surface("instructions", "AGENTS.md", unit="file", free_named=False),
        Surface("skills", "skills", unit="file", write_tools=frozenset({"write", "edit"})),
        Surface("notes", "notes", unit="file", write_tools=frozenset({"write", "edit"})),
        Surface("tools", "tools", unit="file", write_tools=frozenset({"write", "edit"}),
                is_code=True),
        # the harness's real source, exact-synced over the baked tree and rebuilt at boot
        Surface("loop", "src", unit="file", is_code=True, free_named=False,
                write_tools=frozenset({"write", "edit"})),
    )

    def __init__(self, image: str = IMAGE, network: str = "host",
                 provider: str = "deepseek", model: str = "deepseek-v4-flash",
                 key: str | None = None, sandbox=None,
                 phase_timeout_s: int = PHASE_TIMEOUT_S) -> None:
        self.image = image
        self.network = network
        self.provider = provider
        self.model = model
        self.phase_timeout_s = phase_timeout_s
        # per-instance key injection first (multi-tenant runs must not share env)
        self.key = key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_KEY", "")
        from proteus.sandbox import DockerSandbox, SandboxConfig
        # containers write into bind mounts; on Linux a root-in-container write leaves
        # root-owned files the host user can neither snapshot-clean nor edit, so the
        # container runs as the host user (the images chmod their /opt/src for this)
        host_user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else ""
        # a caller may pass its own environment (see DshHarness.__init__)
        self.sandbox = sandbox or DockerSandbox(SandboxConfig(
            network=network, image=image, env_passthrough=("DEEPSEEK_API_KEY",),
            user=host_user,
        ))

    def surfaces(self) -> Sequence[Surface]:
        return self.SURFACES

    def required_edit_tools(self) -> frozenset[str]:
        return frozenset({"write", "edit"})

    def seed(self, harness_root: Path, rng_seed: int = 0) -> None:
        harness_root.mkdir(parents=True, exist_ok=True)
        (harness_root / "AGENTS.md").write_text(SEED_INSTRUCTIONS, encoding="utf-8")
        for sub in ("notes", "tools", "skills"):
            (harness_root / sub).mkdir(exist_ok=True)
        self._extract_self_code(harness_root / "src")

    def _extract_self_code(self, dest: Path) -> None:
        """Unpack the source the image was built from into `dest` (episode-0 state).

        The image bakes a source-only tar at build time precisely so this is cheap and
        exact: what the seed gets is byte-for-byte the source of the build it boots."""
        if dest.exists() and any(dest.iterdir()):
            return                        # resumed root: the seed owns its source already
        dest.mkdir(parents=True, exist_ok=True)
        user = (["--user", f"{os.getuid()}:{os.getgid()}"]
                if hasattr(os, "getuid") else [])
        proc = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", *user,
             "-v", f"{dest}:/proteus-out", "--entrypoint", "sh", self.image,
             "-c", f"tar -xf {SOURCE_TAR} -C /proteus-out --strip-components=1"],
            capture_output=True, text=True, errors="replace", check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"could not extract pi source from {self.image}: {proc.stderr[-300:]}")

    @staticmethod
    def _task_mount(run_root: Path) -> tuple:
        """Bind the run's task workspace (a snapshot-external sibling of the harness)
        into the agent's view, when the run is goal-conditioned."""
        task = run_root / "task"
        return ((str(task), "/workspace/task"),) if task.is_dir() else ()

    def check_boot(self, harness_root: Path) -> str:
        """Viability gate: sync + rebuild + `--version` through the image's boot wrapper.

        Because the wrapper rebuilds from /workspace/src, a type error the agent wrote
        into its own source surfaces here as a legible build failure (exit 97 with the
        build log tail), before any API spend."""
        harness = Path(harness_root)
        state = harness.parent / ".pi-state"
        state.mkdir(exist_ok=True)
        proc = self.sandbox.run(
            harness.parent, ["--version"], env={}, timeout_s=300,
            mounts=((str(harness), "/workspace"), (str(state), "/state")))
        if proc.returncode != 0:
            return (f"self-edited source does not boot (exit {proc.returncode}): "
                    f"{(proc.stderr or proc.stdout)[-400:]}")
        return ""

    def validate_candidate(self, harness_root: Path) -> str:
        """Run the model-free episode-boundary build/boot gate on the candidate."""
        return self.check_boot(harness_root)

    def install_disposition(self, harness_root: Path, disposition: Disposition) -> None:
        instructions.install_block(harness_root / "AGENTS.md", disposition)

    def disposition_fingerprint(self, harness_root: Path) -> str:
        return instructions.block_fingerprint(Path(harness_root) / "AGENTS.md")

    # ------------------------------------------------------------------ episodes

    @staticmethod
    def _sessions(state: Path) -> set[Path]:
        return set(state.glob("*.jsonl"))

    def _session_trace(self, path: Path, phase: str) -> list[ActionEvent]:
        """Normalize one native Pi session for measurement and handoff fallback."""
        events: list[ActionEvent] = []
        turn = 0
        if not path.exists():
            return events
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "message":
                continue
            message = event.get("message", {})
            if message.get("role") != "assistant":
                continue
            turn += 1
            for block in message.get("content", []):
                kind = block.get("type", "")
                if kind in ("toolCall", "tool_call", "toolUse"):
                    args = block.get("arguments") or block.get("input") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    path_arg = str(args.get("file_path") or args.get("path") or "")
                    events.append(ActionEvent(
                        turn=turn, phase=phase, tool=block.get("name", ""),
                        surface=self._surface_for_path(path_arg),
                        params={k: str(v)[:200] for k, v in args.items()}, text="",
                    ))
                elif kind == "text" and block.get("text"):
                    events.append(ActionEvent(
                        turn=turn, phase=phase, tool=None, surface=None,
                        params={}, text=block["text"][:500],
                    ))
        return events

    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        if not self.key:
            return EpisodeResult(episode=spec.episode, ok=False, turns=0,
                                 error="no DeepSeek key: set DEEPSEEK_API_KEY")
        run_root = Path(spec.root)
        harness = run_root / "harness"
        state = run_root / ".pi-state"
        state.mkdir(exist_ok=True)
        handoffs = HandoffStore(run_root)
        (run_root / "traces").mkdir(exist_ok=True)
        mapping: dict[str, list[str]] = {}
        error = ""
        capped = False
        checkpoint_misses = 0
        plan = budget_plan(spec)
        budget = plan.hard_limit
        episode_files: set = set()
        active = Path(spec.active_root) if spec.active_root is not None else harness
        # Core-managed staged episodes already execute a previously validated snapshot.
        # Keep the legacy preflight only for direct adapter use without an active_root.
        if spec.active_root is None and (harness / "src").is_dir():
            error = self.check_boot(harness)
        if spec.active_root is not None:
            # Nested bind targets must exist before Docker mounts /workspace read-only.
            # These placeholders belong to the disposable active copy and are obscured by
            # the writable candidate/framework mounts in the running container.
            (active / "candidate").mkdir(exist_ok=True)
            (active / ".proteus").mkdir(exist_ok=True)
            if (run_root / "task").is_dir():
                (active / "task").mkdir(exist_ok=True)
        workspace_mounts = ((str(active), "/workspace", "ro"),
                            (str(harness), "/workspace/candidate")) \
            if spec.active_root is not None else ((str(harness), "/workspace"),)
        for phase in PHASES if not error else ():
            # the budget is enforced twice, both harness-agnostically: exactly, between
            # phases (no new phase once it is spent) and approximately, mid-phase (the
            # session log is polled and the container stopped at the phase's stop line).
            # BudgetPlan preserves the legacy later-phase reserve or applies the explicit
            # act-priority plan. A phase stop moves on; only the hard ceiling caps the
            # episode.
            used = self._live_calls(state, episode_files, set()) if plan.enabled else 0
            if budget and used >= budget:
                capped = True
                break
            stop_at = plan.stop_at(phase, used)
            if budget and used >= stop_at:
                continue
            handoff_start = handoffs.begin(spec.episode, phase)
            before = self._sessions(state)
            fired = [False]

            def stop_check(before=before, fired=fired, stop_at=stop_at):
                if self._live_calls(state, episode_files,
                                    self._sessions(state) - before) >= stop_at:
                    fired[0] = True
                    return True
                return False

            timed_out = False
            try:
                proc = self.sandbox.run(
                    run_root,
                    ["--provider", self.provider, "--model", spec.model or self.model,
                     "--session-dir", "/state", "--skill", "/workspace/skills",
                     "-p", phase_prompt(spec, phase, used)],
                    env={"DEEPSEEK_API_KEY": self.key},
                    timeout_s=self.phase_timeout_s,
                    mounts=workspace_mounts + ((str(state), "/state"),
                            (str(handoffs.root), CONTAINER_ROOT))
                           + self._task_mount(run_root),
                    stop_check=stop_check if plan.enabled else None,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                proc = None
            new = self._sessions(state) - before
            phase_events: list[ActionEvent] = []
            if new:
                session_paths = sorted(new, key=str)
                mapping[phase] = [p.name for p in session_paths]
                episode_files |= new
                for session_path in session_paths:
                    phase_events.extend(self._session_trace(session_path, phase))
            handoff = handoffs.finish(handoff_start, phase_events,
                                      interrupted=timed_out or fired[0])
            if spec.checkpoint_turns and handoff["source"] != "agent":
                checkpoint_misses += 1
            if timed_out:
                error = f"phase {phase}: timeout after {self.phase_timeout_s}s"
                break
            assert proc is not None
            if proc.returncode != 0:
                if fired[0]:
                    # stopped at the phase's line: continue if it was only the reserve,
                    # end the episode only when the whole budget is spent
                    if budget and self._live_calls(state, episode_files, set()) >= budget:
                        capped = True
                        break
                    continue
                error = f"phase {phase}: exit {proc.returncode}: {proc.stderr[-400:]}"
                break
        (run_root / "traces" / f"ep{spec.episode:03d}.json").write_text(
            json.dumps(mapping, indent=1))
        trace = self.read_trace(run_root, spec.episode)
        phase_counts = {
            phase: sum(1 for event in trace if event.phase == phase and event.tool)
            for phase in PHASES
        }
        counters = {"phases": len(mapping), "turn_capped": capped,
                    "checkpoint_misses": checkpoint_misses}
        counters.update({f"phase_{phase}_turns": count
                         for phase, count in phase_counts.items()})
        return EpisodeResult(
            episode=spec.episode, ok=not error,
            turns=sum(1 for e in trace if e.tool), error=error,
            counters=counters,
        )

    def _live_calls(self, state: Path, episode_files: set, extra: set) -> int:
        """Tool calls made so far this episode, read live from the session logs.

        pi's session JSONL is plain text and appended per event, so a mid-phase read
        sees every call already made. Counted by marker substring — over-counting stops
        early, which is the conservative direction for a budget."""
        n = 0
        for f in set(episode_files) | set(extra):
            path = Path(f) if isinstance(f, Path) else state / f
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                n += sum(text.count(f'"{marker}"')
                         for marker in ("toolCall", "tool_call", "toolUse"))
            except OSError:
                continue
        return n

    # ------------------------------------------------------------------ measure path

    def _surface_for_path(self, file_path: str) -> Optional[str]:
        p = file_path
        for prefix in ("/workspace/candidate/", "/workspace/", "candidate/"):
            if p.startswith(prefix):
                p = p[len(prefix):]
                break
        if p == "AGENTS.md":
            return "instructions"
        for s in ("skills", "notes", "tools"):
            if p.startswith(f"{s}/"):
                return s
        if p.startswith("src/"):
            return "loop"
        return None

    def read_trace(self, root: Path, episode: int) -> Sequence[ActionEvent]:
        root = Path(root)
        map_path = root / "traces" / f"ep{episode:03d}.json"
        if not map_path.exists():
            return []
        mapping = json.loads(map_path.read_text())
        state = root / ".pi-state"
        events: list[ActionEvent] = []
        turn = 0
        for phase in PHASES:
            names = mapping.get(phase)
            if not names:
                continue
            if isinstance(names, str):
                names = [names]                # traces written before the list format
            for name in names:
                if not (state / name).exists():
                    continue
                phase_events = self._session_trace(state / name, phase)
                for event in phase_events:
                    events.append(ActionEvent(
                        turn=turn + event.turn, phase=event.phase, tool=event.tool,
                        surface=event.surface, params=event.params, text=event.text,
                    ))
                turn += max((event.turn for event in phase_events), default=0)
        return events
