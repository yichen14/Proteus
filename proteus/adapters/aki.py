"""Aki adapter — the default, full-featured research harness.

The reference integration: the Aki harness (persistent memory + writable skills +
self-authored tools + an editable episode loop) plugged into Proteus. It is the harness the
paper's headline experiments run on.

Two paths, deliberately separated:

- **Measure path** (`read_trace`, `disposition_fingerprint`): pure JSONL/file parsing. Needs
  no Aki checkout at all, so Proteus's rulers read existing Aki run roots as-is.
- **Run path** (`seed`, `install_disposition`, `run_episode`): drives the Aki research
  runner. Needs the checkout (`AKI_HARNESS_SRC` or the `src=` argument) and imports it
  lazily. Episodes launch through Aki's own supervisor, containerized when `docker=True`
  (an authored Aki tool executes arbitrary in-process code, so OS-level isolation is the
  only real boundary).

The adapter never writes into the Aki checkout; all state lands in the Proteus run root.

Known limits, stated rather than papered over:
- Aki's phase prompts live inside the harness (`loop.py`), so `spec.phase_prompts` is not
  injected into the episode yet; no-goal runs — the paper's primary regime — are fully
  supported, while the framework default epistemic protocol and goal-text injection are
  not wired into Aki episodes. Other bundled adapters consume `spec.phase_prompts`.
- Aki couples seeding and disposition install in one `init_run`, so this adapter performs
  both inside `install_disposition` (the framework calls `seed` first; it only records state).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from proteus.core.adapter import ActionEvent, EpisodeResult, EpisodeSpec, Surface
from proteus.core.disposition import Disposition

#: Aki phase names -> Proteus phase names.
_PHASE = {"observe": "observe", "propose": "propose",
          "select_and_act": "act", "reflect": "reflect"}

#: Proteus disposition -> Aki grid arm label.
_ARM = {
    ("neutral", None): "C0_neutral",
    ("review", "memory"): "Cmem_memory",
    ("review", "skills"): "Cskl_skills",
    ("review", "tools"): "Ctol_tools",
    ("record", "memory"): "L4mem_memory",
    ("record", "skills"): "L4skl_skills",
    ("record", "tools"): "L4tol_tools",
}


class AkiHarness:
    """`HarnessAdapter` for the Aki research harness."""

    name = "aki"
    continuity_mode = "native"     # Aki's supervisor owns its internal phase state
    disposition_in_files = True    # the apparatus installs the persona through its carrier

    SURFACES = (
        Surface("memory", "memory", unit="file", write_tools=frozenset({"memory_write"})),
        Surface("skills", "skills", unit="directory",
                write_tools=frozenset({"skill_write", "skill_update"})),
        Surface("tools", "tools", unit="file", write_tools=frozenset({"tool_write"}),
                is_code=True),
        Surface("loop", "loop.py", unit="top_level_def", is_code=True, free_named=False,
                write_tools=frozenset({"file_write", "file_edit"})),
    )

    def __init__(self, src: str | None = None, docker: bool = True) -> None:
        self.src = Path(src or os.environ.get("AKI_HARNESS_SRC", "")).expanduser()
        self.docker = docker
        self._pending_root: Optional[Path] = None
        self._run_configs: dict[Path, object] = {}

    # ---------------------------------------------------------------- contract: metadata

    def surfaces(self) -> Sequence[Surface]:
        return self.SURFACES

    def required_edit_tools(self) -> frozenset[str]:
        return frozenset({"memory_write", "skill_write", "tool_write", "file_write"})

    # ---------------------------------------------------------------- run path (needs Aki)

    def _api(self):
        """Import the Aki research runner, lazily and only for the run path."""
        if not self.src or not self.src.exists():
            raise RuntimeError(
                "Running Aki episodes needs the Aki checkout: set AKI_HARNESS_SRC or pass "
                "AkiHarness(src=...). Reading/measuring existing Aki runs needs neither."
            )
        # The supervisor reads its containment flag at import time.
        os.environ["AKI_EPISODE_DOCKER"] = "1" if self.docker else "0"
        if str(self.src) not in sys.path:
            sys.path.insert(0, str(self.src))
        from experiments import grid
        from experiments.runner import supervisor
        from experiments.runner.config import Condition, RunConfig
        return grid, supervisor, Condition, RunConfig

    def _arm_label(self, disposition: Disposition) -> str:
        if "AKI_ARM" in disposition.config:
            return disposition.config["AKI_ARM"]
        if disposition.is_empty:
            return _ARM[("neutral", None)]
        kind = disposition.label.split("_", 1)[0]
        surface = disposition.config.get("SURFACE")
        try:
            return _ARM[(kind, surface)]
        except KeyError:
            raise ValueError(
                f"no Aki arm for disposition {disposition.label!r}; pass an explicit arm "
                "via Disposition.config['AKI_ARM']"
            ) from None

    def seed(self, harness_root: Path, rng_seed: int = 0) -> None:
        # Aki's init_run seeds and installs in one step; remember state until then.
        self._pending_root = harness_root
        self._rng_seed = rng_seed

    def install_disposition(self, harness_root: Path, disposition: Disposition) -> None:
        grid, supervisor, Condition, RunConfig = self._api()
        arm = self._arm_label(disposition)
        # Materialize the arm's persona under the Proteus sweep root, not the research
        # grid's own root — a concurrently running grid must never see our writes.
        grid.ROOT = harness_root.parent.parent
        grid.ROOT.mkdir(parents=True, exist_ok=True)
        grid._install_arm(arm)
        run_root = harness_root.parent
        config = RunConfig(
            condition=Condition(None, None, label=arm),
            seed=getattr(self, "_rng_seed", int(os.environ.get("PROTEUS_AKI_SEED", "0"))),
            episodes=1,  # the Proteus framework owns the loop; per-call episode below
            root=run_root,
        )
        if not harness_root.exists():
            supervisor.init_run(config)
        self._run_configs[run_root] = config
        self._pending_root = None

    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        _, supervisor, _, _ = self._api()
        run_root = Path(spec.root)
        config = self._run_configs.get(run_root)
        if config is None:
            raise RuntimeError("run_episode before install_disposition for this root")
        asyncio.run(supervisor.run_episode(config, spec.episode))
        # Read the outcome back from the trace — the only channel Proteus trusts.
        status, counters = self._episode_outcome(run_root, spec.episode)
        return EpisodeResult(
            episode=spec.episode,
            ok=status.get("status") == "complete",
            turns=int(counters.get("turns_used", 0)),
            error=str(status.get("error", "")),
            counters={k: v for k, v in counters.items() if isinstance(v, (int, float))},
        )

    # ---------------------------------------------------------------- measure path (no Aki)

    @staticmethod
    def _trace_path(root: Path, episode: int) -> Path:
        traces = Path(root) / "traces"
        for candidate in (traces / f"ep{episode:03d}.jsonl", traces / f"ep{episode}.jsonl"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no trace for episode {episode} under {traces}")

    def _surface_for_tool(self, tool: str) -> Optional[str]:
        for s in self.SURFACES:
            if tool in s.write_tools:
                return s.name
        return None

    def _episode_outcome(self, root: Path, episode: int) -> tuple[dict, dict]:
        status: dict = {}
        counters: dict = {}
        for event in self._events(root, episode):
            if event.get("event") == "episode_status":
                status = event.get("data", {})
            elif event.get("event") == "episode_end":
                counters = event.get("data", {}).get("counters", {})
        return status, counters

    def _events(self, root: Path, episode: int):
        path = self._trace_path(root, episode)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def read_trace(self, root: Path, episode: int) -> Sequence[ActionEvent]:
        events: list[ActionEvent] = []
        phase = "observe"
        for e in self._events(root, episode):
            kind = e.get("event")
            data = e.get("data", {})
            if kind == "phase_start":
                phase = _PHASE.get(data.get("phase", ""), data.get("phase", ""))
            elif kind == "tool_call":
                tool = data.get("tool_name", "")
                events.append(ActionEvent(
                    turn=int(e.get("iteration", 0)), phase=phase, tool=tool,
                    surface=self._surface_for_tool(tool),
                    params=data.get("params", {}) or {}, text="",
                ))
            elif kind == "reply":
                events.append(ActionEvent(
                    turn=int(e.get("iteration", 0)), phase=phase, tool=None,
                    surface=None, params={}, text=str(data.get("content", "")),
                ))
        return events

    def disposition_fingerprint(self, harness_root: Path) -> str:
        """Hash the installed-disposition carriers so drift of F is detectable."""
        harness_root = Path(harness_root)
        h = hashlib.sha256()
        candidates = sorted(
            [harness_root / "loop.py",
             *(harness_root.parent / ".aki").glob("persona*"),
             *harness_root.glob("disposition_*.py")], key=str)
        for p in candidates:
            if p.is_file():
                h.update(p.name.encode())
                h.update(p.read_bytes())
        return h.hexdigest()[:16]
