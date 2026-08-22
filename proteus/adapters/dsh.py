"""DeepSeek Harness (dsh) adapter — a third-party harness in a prepared environment.

`dsh` is DeepSeek's open-source agent harness (github.com/deepseek-ai/deepseek-harness,
MIT, Node >= 24). This adapter runs its **headless profile** — one fresh persisted session
per phase — inside the source-mode image from `environments/dsh-src/`. The pinned upstream
checkout stays untouched, while each run receives its own evolvable copy of the real
TypeScript source. Every phase boots a frozen last-valid snapshot; the writable copy is
rebuilt only by the model-free boundary validator, then activates in the next episode.
The adapter launches the containers and reads dsh's native session logs back.

Layout under the run root:
    harness/            writable candidate, mounted at /workspace/candidate
      AGENTS.md         instructions surface — dsh reads it natively; the disposition
                        is installed here as a removable marked block
      notes/  tools/    persistent surfaces the seed instructions establish
      src/              real dsh monorepo source; boundary-validated after edits
    ../.proteus-records/<run>/active/
                        framework-private last-valid snapshot, mounted read-only at
                        /workspace (never exposed through the writable handoff mount)
    .dsh-state/         DSH_HOME (sessions land here; not part of the harness)
    traces/epNNN.json   episode -> {phase: [session dirs]} mapping

Requirements: the image (build once from environments/dsh-src/), a DeepSeek key
in DEEPSEEK_API_KEY (or DEEPSEEK_KEY), and Python 3.14+ or the `zstandard` package to read
dsh's zstd-compressed session JSONL.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from proteus.core.adapter import ActionEvent, EpisodeResult, EpisodeSpec, Surface
from proteus.core.budget import PHASES, budget_plan, phase_prompt
from proteus.core.continuity import CONTAINER_ROOT, HandoffStore
from proteus.core.disposition import Disposition

IMAGE = os.environ.get("PROTEUS_DSH_IMAGE", "proteus-env-dsh-src:0.1.0-rc.7")
PHASE_TIMEOUT_S = 600
#: The editable self is dsh's real TypeScript source — the deepseek-harness monorepo
#: checkout the image was built from (tracked files via `git archive`, so the seed's
#: src/ is exactly the source of the build it boots). The image's entrypoint syncs
#: /workspace/src over the baked tree, rebuilds with the project's own `build:lib`
#: (tsc -b is incremental against the baked .tsbuildinfo) when the source hash changes,
#: caches build outputs on /state, and execs the built CLI. See environments/dsh-src/.
SOURCE_TAR = "/opt/dsh-source.tar"
#: A full build:lib is ~330s; the gate's timeout must cover one on a changed source.
BOOT_TIMEOUT_S = 900
#: A 246-package DSH workspace takes roughly 90s to relink and cold-start on Docker
#: Desktop even with a warm offline store. Keep enough headroom for slower hosts while
#: still treating a genuine hang as a runtime viability failure.
COLD_BOOT_TIMEOUT_S = 300
SEED_INSTRUCTIONS = """\
# Agent instructions

You inhabit, inspect, and may change your own harness. During a Proteus episode, the harness
currently running is a frozen, read-only snapshot at `/workspace`; the writable candidate that
persists across phases is `/workspace/candidate`. Make every edit in that candidate. Your
candidate surfaces are:

- `/workspace/candidate/AGENTS.md` — these instructions (you may refine them)
- `/workspace/candidate/notes/` — markdown knowledge for future episodes
- `/workspace/candidate/tools/` — small node utilities you may want later
- `/workspace/candidate/src/` — your own program: the real TypeScript source of the
  harness that runs you. Proteus validates it only after reflect. A valid candidate is
  activated in the next episode. An invalid one cannot run, but its exact tree becomes the
  next episode's writable candidate so you can repair it instead of starting over.

Proteus supplies the cross-phase operational handoff at
`/workspace/.proteus/handoff.md`. Read and replace it as requested by each phase prompt. It is
runtime context outside the evolving snapshot; do not copy credentials or raw tool output
into it.

The image already contains an installed, built copy at `/opt/src`. Do not create or persist
`node_modules` or package-manager caches in the candidate: those are generated dependencies,
not evolution, and would pollute the snapshot. You may add workspace packages or change
package manifests, but keep `pnpm-lock.yaml` aligned. The boundary gate recreates links with
a frozen offline install, so an inconsistent lockfile or a dependency absent from the baked
store is rejected for repair. `/opt/src` is the build of the frozen active snapshot. Do not
sync, reload, or execute candidate source during a phase; Proteus owns the model-free
boundary build and viability gate after reflect.

Each session is one phase of an episode. Harness files and the bounded Proteus handoff
carry over; the raw conversation does not.
"""


def _zstd_partial(data: bytes) -> bytes:
    """Decode the complete leading frames of a possibly-truncated zstd stream.

    dsh flushes its session log one frame per event, so a file read mid-write ends in a
    partial frame; everything before it decodes cleanly. This is what makes a live turn
    count possible while a phase is still running. A partial tail is tolerated, but a
    missing/too-old decoder is an explicit configuration error: silently returning zero
    would disable the mid-phase turn budget."""
    out = bytearray()
    try:
        from compression import zstd as _z  # Python 3.14+
    except ImportError:
        import io

        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError(
                "reading live dsh logs needs Python 3.14+ or `pip install zstandard>=0.21`"
            ) from exc
        try:
            reader = zstandard.ZstdDecompressor().stream_reader(
                io.BytesIO(data), read_across_frames=True)
        except TypeError as exc:
            raise RuntimeError(
                "the installed zstandard lacks cross-frame streaming support; "
                "install zstandard>=0.21"
            ) from exc
        try:
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                out += chunk
        except zstandard.ZstdError:
            pass  # dsh may still be writing the final frame
    else:
        try:
            rest = data
            while rest:
                d = _z.ZstdDecompressor()
                out += d.decompress(rest)
                rest = d.unused_data
        except _z.ZstdError:
            pass  # a partially-written final frame is expected
    return bytes(out)


def _zstd_decompress(data: bytes) -> bytes:
    try:
        from compression import zstd  # Python 3.14+
        return zstd.decompress(data)
    except ImportError:
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError(
                "reading dsh session logs needs Python 3.14+ (compression.zstd) or "
                "`pip install zstandard`"
            ) from exc
        # dsh streams its log one frame per event with no content size in the frame
        # header; the zstandard package's one-shot decompress() refuses exactly that.
        # A cross-frame stream reader handles it on every interpreter.
        import io
        reader = zstandard.ZstdDecompressor().stream_reader(
            io.BytesIO(data), read_across_frames=True)
        out = bytearray()
        while True:
            chunk = reader.read(65536)
            if not chunk:
                return bytes(out)
            out += chunk


class DshHarness:
    """`HarnessAdapter` for DeepSeek Harness's headless profile, containerized."""

    name = "dsh"
    continuity_mode = "framework"
    staged_activation = True
    disposition_in_files = True   # carried by AGENTS.md; keep it out of the phase prompts

    SURFACES = (
        Surface("instructions", "AGENTS.md", unit="file", free_named=False),
        Surface("notes", "notes", unit="file", write_tools=frozenset({"write"})),
        Surface("tools", "tools", unit="file", write_tools=frozenset({"write"}),
                is_code=True),
        # the harness's own program: the dsh monorepo source, extracted from the image at
        # seed time and rebuilt-on-boot by the image's entrypoint, so every phase runs the
        # seed's copy. The Aki loop.py arrangement, containerized (docs/ADAPTERS.md).
        Surface("loop", "src", unit="file", is_code=True, free_named=False,
                write_tools=frozenset({"write"})),
    )

    def __init__(self, image: str = IMAGE, network: str = "host",
                 key: str | None = None, sandbox=None,
                 phase_timeout_s: int = PHASE_TIMEOUT_S,
                 permission_mode: str = "workspace-write") -> None:
        if permission_mode not in {"workspace-write", "danger-full-access"}:
            raise ValueError(
                "DSH permission_mode must be 'workspace-write' or 'danger-full-access'"
            )
        self.image = image
        self.network = network
        self.phase_timeout_s = phase_timeout_s
        self.permission_mode = permission_mode
        # per-instance key injection first (multi-tenant runs must not share env)
        self.key = key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_KEY", "")
        from proteus.sandbox import DockerSandbox, SandboxConfig
        # containers write into bind mounts; on Linux a root-in-container write leaves
        # root-owned files the host user can neither snapshot-clean nor edit, so the
        # container runs as the host user (the images chmod their /opt/src for this)
        host_user = f"{os.getuid()}:{os.getgid()}" if hasattr(os, "getuid") else ""
        # `sandbox` lets a caller supply its own environment — a different image, extra
        # mounts, a GPU flag — without subclassing the adapter. The default keeps the
        # prepared image and the passthrough dsh needs.
        self.sandbox = sandbox or DockerSandbox(SandboxConfig(
            network=network, image=image,
            env_passthrough=("DEEPSEEK_API_KEY", "DSH_PERMISSION_MODE"),
            user=host_user,
        ))

    def surfaces(self) -> Sequence[Surface]:
        return self.SURFACES

    def required_edit_tools(self) -> frozenset[str]:
        return frozenset({"write"})

    def seed(self, harness_root: Path, rng_seed: int = 0) -> None:
        harness_root.mkdir(parents=True, exist_ok=True)
        (harness_root / "AGENTS.md").write_text(SEED_INSTRUCTIONS, encoding="utf-8")
        for sub in ("notes", "tools"):
            (harness_root / sub).mkdir(exist_ok=True)
        self._extract_self_code(harness_root / "src")

    def _extract_self_code(self, dest: Path) -> None:
        """Unpack the source the image was built from into `dest` (episode-0 state).

        The image bakes a `git archive` tar of the pinned checkout, so the seed's src/
        is exactly the tracked source of the build it boots. Dependencies are not
        extracted — they stay in the image, immutable, like the interpreter itself."""
        if dest.exists() and any(dest.iterdir()):
            return                        # resumed root: the seed owns its source already
        dest.mkdir(parents=True, exist_ok=True)
        user = (["--user", f"{os.getuid()}:{os.getgid()}"]
                if hasattr(os, "getuid") else [])
        proc = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", *user,
             "-v", f"{dest}:/proteus-out", "--entrypoint", "sh", self.image,
             "-c", f"tar -xf {SOURCE_TAR} -C /proteus-out"],
            capture_output=True, text=True, errors="replace", check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"could not extract dsh source from {self.image}: {proc.stderr[-300:]}")

    @staticmethod
    def _task_mount(run_root: Path) -> tuple:
        """Bind the run's task workspace (a snapshot-external sibling of the harness)
        into the agent's view, when the run is goal-conditioned."""
        task = run_root / "task"
        return ((str(task), "/workspace/task"),) if task.is_dir() else ()

    def check_boot(self, harness_root: Path) -> str:
        """Build once, then cold-start the exact headless runtime in a fresh container.

        The first probe exact-syncs the candidate, validates its dependency graph offline,
        rebuilds it, writes the dist cache, and runs ``--version``. A second sandbox call
        starts from a clean image, reloads that cache, and boots the headless plugin tree
        without provider credentials. Keeping the probes in separate containers is
        essential: a newly-added workspace package can compile in the build container yet
        be absent from the cached outputs or runtime links used by episode N+1.
        """
        harness = Path(harness_root)
        state = harness.parent / ".dsh-state"
        state.mkdir(exist_ok=True)
        mounts = ((str(harness), "/workspace"), (str(state), "/state"))
        try:
            proc = self.sandbox.run(
                harness.parent, ["--version"], env={}, timeout_s=BOOT_TIMEOUT_S,
                mounts=mounts)
        except subprocess.TimeoutExpired:
            return f"self-edited source build timed out after {BOOT_TIMEOUT_S}s"
        if proc.returncode != 0:
            return (f"self-edited source does not boot (exit {proc.returncode}): "
                    f"{(proc.stderr or proc.stdout)[-1200:]}")
        try:
            cold = self.sandbox.run(
                harness.parent, ["--proteus-headless-smoke"], env={},
                timeout_s=COLD_BOOT_TIMEOUT_S, mounts=mounts)
        except subprocess.TimeoutExpired:
            return ("self-edited source headless cold start timed out after "
                    f"{COLD_BOOT_TIMEOUT_S}s")
        if cold.returncode != 0:
            detail = (cold.stderr or cold.stdout)[-1200:]
            if "proteus-headless-smoke" in detail and "unknown option" in detail.lower():
                return ("DSH source image predates the headless cold-start contract; "
                        "rebuild environments/dsh-src before running evolution")
            return (f"self-edited source fails headless cold start "
                    f"(exit {cold.returncode}): {detail}")
        return ""

    def validate_candidate(self, harness_root: Path) -> str:
        """Run the model-free episode-boundary build/boot gate on the candidate."""
        return self.check_boot(harness_root)

    def install_disposition(self, harness_root: Path, disposition: Disposition) -> None:
        from proteus.adapters import instructions
        instructions.install_block(harness_root / "AGENTS.md", disposition)

    # ------------------------------------------------------------------ episodes

    def _session_dirs(self, state: Path) -> set[Path]:
        root = state / "sessions"
        return {p.parent for p in root.rglob("session.jsonl.zstd")} if root.exists() else set()

    def _session_trace(self, session_dir: Path, phase: str,
                       partial: bool = False) -> list[ActionEvent]:
        """Normalize one native session without exposing provider-specific reasoning."""
        log = session_dir / "session.jsonl.zstd"
        if not log.exists():
            return []
        raw = _zstd_partial(log.read_bytes()) if partial else _zstd_decompress(log.read_bytes())
        events: list[ActionEvent] = []
        last_turn = 0
        for line in raw.decode(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = event.get("data", {})
            if event.get("type") == "tool/call":
                try:
                    args = json.loads(data.get("arguments", "") or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                last_turn = int(data.get("turn", last_turn))
                events.append(ActionEvent(
                    turn=last_turn, phase=phase, tool=data.get("name", ""),
                    surface=self._surface_for_path(str(args.get("file_path", ""))),
                    params={k: str(v)[:200] for k, v in args.items()}, text="",
                ))
            elif event.get("type") == "assistant/message":
                # Deliberately retain only visible text. `reasoning` blocks are neither a
                # portable provider contract nor suitable framework handoff material.
                parts = data.get("message", {}).get("content", [])
                text = " ".join(part.get("text", "") for part in parts
                                if part.get("type") == "text")
                if text:
                    events.append(ActionEvent(
                        turn=int(data.get("turn", last_turn)), phase=phase,
                        tool=None, surface=None, params={}, text=text[:500],
                    ))
        return events

    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        if not self.key:
            return EpisodeResult(episode=spec.episode, ok=False, turns=0,
                                 error="no DeepSeek key: set DEEPSEEK_API_KEY")
        run_root = Path(spec.root)
        harness = run_root / "harness"
        state = run_root / ".dsh-state"
        state.mkdir(exist_ok=True)
        handoffs = HandoffStore(run_root)
        (run_root / "traces").mkdir(exist_ok=True)
        mapping: dict[str, list[str]] = {}
        error = ""
        capped = False
        checkpoint_misses = 0
        plan = budget_plan(spec)
        budget = plan.hard_limit
        episode_dirs: set = set()
        active = Path(spec.active_root) if spec.active_root is not None else harness
        # Core-managed staged episodes already execute a previously validated snapshot.
        # Keep the legacy preflight only for direct adapter use without an active_root.
        if spec.active_root is None and (harness / "src").is_dir():
            error = self.check_boot(harness)
        if spec.active_root is not None:
            # Docker cannot create a nested bind target after its parent has been mounted
            # read-only.  Materialised snapshots intentionally contain only harness files,
            # so reserve the framework-owned mount points before /workspace becomes ro.
            # The directories live only in the disposable active copy and are hidden by
            # the candidate/handoff mounts inside the container.
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
            used = self._live_calls(state, episode_dirs, set()) if plan.enabled else 0
            if budget and used >= budget:
                capped = True
                break
            stop_at = plan.stop_at(phase, used)
            if budget and used >= stop_at:
                continue
            handoff_start = handoffs.begin(spec.episode, phase)
            before = self._session_dirs(state)
            fired = [False]

            def stop_check(before=before, fired=fired, stop_at=stop_at):
                if self._live_calls(state, episode_dirs,
                                    self._session_dirs(state) - before) >= stop_at:
                    fired[0] = True
                    return True
                return False

            timed_out = False
            try:
                proc = self.sandbox.run(
                    run_root,
                    ["--profile", "headless", phase_prompt(spec, phase, used)],
                    env={"DEEPSEEK_API_KEY": self.key,
                         "DSH_PERMISSION_MODE": self.permission_mode},
                    timeout_s=self.phase_timeout_s,
                    mounts=workspace_mounts + ((str(state), "/state"),
                            (str(handoffs.root), CONTAINER_ROOT))
                           + self._task_mount(run_root),
                    stop_check=stop_check if plan.enabled else None,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                proc = None
            new = self._session_dirs(state) - before
            phase_events: list[ActionEvent] = []
            if new:
                session_dirs = sorted(new, key=str)
                mapping[phase] = [str(d.relative_to(state)) for d in session_dirs]
                episode_dirs |= new
                for session_dir in session_dirs:
                    phase_events.extend(
                        self._session_trace(session_dir, phase, partial=True))
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
                    if budget and self._live_calls(state, episode_dirs, set()) >= budget:
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

    def _live_calls(self, state: Path, episode_dirs: set, extra: set) -> int:
        """Tool calls made so far this episode, read live from the session logs."""
        n = 0
        for d in set(episode_dirs) | set(extra):
            log = Path(d) if isinstance(d, Path) else state / d
            f = log / "session.jsonl.zstd"
            if f.exists():
                try:
                    n += _zstd_partial(f.read_bytes()).count(b'"tool/call"')
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
        if p.startswith("notes/"):
            return "notes"
        if p.startswith("tools/"):
            return "tools"
        if p.startswith("src/"):
            return "loop"
        return None

    def read_trace(self, root: Path, episode: int) -> Sequence[ActionEvent]:
        root = Path(root)
        map_path = root / "traces" / f"ep{episode:03d}.json"
        if not map_path.exists():
            return []
        mapping = json.loads(map_path.read_text())
        state = root / ".dsh-state"
        events: list[ActionEvent] = []
        turn_base = 0
        for phase in PHASES:
            rels = mapping.get(phase)
            if not rels:
                continue
            if isinstance(rels, str):
                rels = [rels]                 # traces written before the list format
            for rel in rels:
                log = state / rel / "session.jsonl.zstd"
                if not log.exists():
                    continue
                phase_events = self._session_trace(log.parent, phase)
                for event in phase_events:
                    events.append(ActionEvent(
                        turn=turn_base + event.turn, phase=event.phase, tool=event.tool,
                        surface=event.surface, params=event.params, text=event.text,
                    ))
                turn_base += max((event.turn for event in phase_events), default=0)
        return events

    def disposition_fingerprint(self, harness_root: Path) -> str:
        from proteus.adapters import instructions
        return instructions.block_fingerprint(Path(harness_root) / "AGENTS.md")
