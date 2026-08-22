"""The harness-agnostic contract.

Proteus drives self-evolution over an *arbitrary* agent harness. It knows nothing about
any specific harness (Aki, OpenHands, SWE-agent, your own loop); it talks only to a
`HarnessAdapter`. A harness ships one adapter and Proteus can then evolve it, measure it,
and swap dispositions in and out of it uniformly.

The three things a self-evolution framework needs from a harness, and none of the existing
agent frameworks expose together, are declared here:

  1. its persistent, editable **surfaces** (memory / skills / tools / code / ...), as data
     so the measurement layer iterates a manifest instead of hard-coded names;
  2. a way to **run one episode** and emit a normalized **action trace**;
  3. a way to **install and remove a disposition** (the action-preference perturbation)
     and to **snapshot** the working state, so divergence and crystallization are
     measurable.

Everything else (which model, how the loop is written, what the tools are) is the
harness's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Surface:
    """One persistent, agent-editable region of the harness.

    A surface is the unit the measurement layer counts over. `memory`, `skills`, and
    `tools` are the familiar ones, but a harness declares its own — a bare ReAct loop
    might expose only `notes`; a coding agent might add `patches`. Because the set is
    data, not a constant baked into the framework, any harness is measurable.
    """

    name: str
    subdir: str                      # directory under the harness root holding this surface
    unit: str = "file"               # "file" | "directory" | "top_level_def"
    write_tools: frozenset[str] = frozenset()   # tool names that count as edits here
    is_code: bool = False            # edits pass the viability gate (must still run)
    free_named: bool = True          # agent chooses unit names (affects rename-matching)


@dataclass(frozen=True)
class ActionEvent:
    """One normalized step of the action trace.

    The trace is the only channel Proteus reads to characterise behaviour — never the
    agent's self-report. A harness emits these (or Proteus derives them from the harness's
    own log via `read_trace`), so measurement never depends on any harness's internal
    event taxonomy.
    """

    turn: int
    phase: str                       # observe | propose | act | reflect
    tool: str | None = None          # None => a text-only turn (a decision to reason/stop)
    surface: str | None = None       # which surface this call targeted, if any
    params: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""                   # the model's text for this step, if any


@dataclass(frozen=True)
class EpisodeSpec:
    """What the harness needs to run exactly one context-fresh episode."""

    root: Path                       # the run root; harness working tree lives at root/harness
    episode: int                     # 1-based
    model: str                       # model id the harness should use
    phase_prompts: Mapping[str, str] # observe/propose/act/reflect texts (default episode
                                     # protocol, goal, and visible evaluator feedback
                                     # already merged in by the framework)
    max_turns: int = 100
    min_turns_per_phase: int = 0
    """Reserve at least this many turns for each phase. While phase i runs, its stop
    line is `max_turns - min_turns_per_phase * phases_remaining_after_i`: a phase that
    reaches its line is ended early so the later phases keep their reserve. A
    reservation stop moves to the next phase; only a spent budget ends the episode."""
    phase_turns: Mapping[str, int] = field(default_factory=dict)
    """Optional explicit normal-plan allocation for observe/propose/act/reflect. When
    present it replaces ``min_turns_per_phase`` and must sum to ``max_turns``."""
    hard_max_turns: int = 0
    """Optional burst ceiling for an explicit phase plan. Zero means ``max_turns``."""
    checkpoint_turns: int = 0
    """Final calls in each explicit phase allowance reserved by the prompt protocol for
    a persistent continuation checkpoint."""
    announce_budget: bool = False
    """Whether the adapter should show the live budget contract at phase start."""
    seed: int = 0                    # the run's RNG seed (condition replicate index)
    extra_env: Mapping[str, str] = field(default_factory=dict)
    continuity_mode: str = "native"
    """How fresh phases continue: ``native`` means the harness owns continuity,
    ``framework`` uses Proteus's external handoff protocol, and ``none`` deliberately
    leaves phases independent. The adapter declaration is copied here by the core."""
    active_root: Path | None = None
    """Frozen harness snapshot that must execute every phase of this episode.

    `root/harness` remains the writable candidate. Adapters declaring
    `staged_activation = True` execute from `active_root` while exposing the candidate
    separately for edits. Candidate changes may be inspected, but only the model-free
    episode-boundary validator may execute them; they never become the controlling harness
    until the next episode.
    """


@runtime_checkable
class HarnessAdapter(Protocol):
    """The whole contract. Implement this and Proteus can evolve your harness."""

    name: str

    # Adapters may declare `continuity_mode = "framework" | "none"`. It is deliberately
    # optional rather than a Protocol member so existing custom adapters stay compatible;
    # absence means `native`. Framework adapters mount the run's external
    # `.proteus-state` at `/workspace/.proteus` and use HandoffStore between phases.

    # Adapters whose own editable files can affect execution may declare
    # `staged_activation = True`. The core then materializes a frozen active snapshot
    # for EpisodeSpec.active_root. Such adapters must keep that runtime separate from
    # the writable candidate and may expose `validate_candidate(harness_root) -> str`.

    disposition_in_files: bool = False
    """True when `install_disposition` writes the perturbation into a file the harness
    loads on its own (an `AGENTS.md`-style instructions surface).

    The framework otherwise appends the disposition text to every phase prompt. For a
    file-carrying harness that would deliver the same text twice per phase — roughly
    double the intended dose, through two channels of different salience — and the
    prompt copy is a transient injection that lives outside `F`, so it is neither
    removable nor covered by `disposition_fingerprint`. Declaring True keeps the single
    planted difference in `F`, which is what makes later divergence attributable.
    """

    # --- static declaration -------------------------------------------------------------
    def surfaces(self) -> Sequence[Surface]:
        """The editable surfaces this harness exposes, as a manifest."""
        ...

    def required_edit_tools(self) -> frozenset[str]:
        """Tools whose presence proves the harness can still edit itself (viability)."""
        ...

    # --- provisioning -------------------------------------------------------------------
    def seed(self, harness_root: Path, rng_seed: int = 0) -> None:
        """Materialise a fresh copy of the harness into `harness_root` (episode 0 state).
        `rng_seed` is the run's replicate index, available to harnesses that need it at
        provisioning time (e.g. for deterministic per-seed naming)."""
        ...

    def install_disposition(self, harness_root: Path, disposition: "Disposition") -> None:
        """Apply the action-preference perturbation. Must be removable (see `Disposition`)."""
        ...

    # --- run + observe ------------------------------------------------------------------
    def run_episode(self, spec: EpisodeSpec) -> "EpisodeResult":
        """Run one episode against the harness at `spec.root`. Returns the result; the
        action trace is obtained via `read_trace` so the loop stays print-safe."""
        ...

    def read_trace(self, root: Path, episode: int) -> Sequence[ActionEvent]:
        """The normalized action trace for one episode."""
        ...

    # --- immutability observable --------------------------------------------------------
    def disposition_fingerprint(self, harness_root: Path) -> str:
        """A hash of the currently-installed disposition, to detect drift of F over time."""
        ...


@dataclass
class EpisodeResult:
    """What `run_episode` reports. Results flow through the trace, not stdout."""

    episode: int
    ok: bool
    turns: int = 0
    error: str = ""
    counters: Mapping[str, Any] = field(default_factory=dict)


# `Disposition` is imported lazily to avoid a cycle; declared here for the Protocol only.
from proteus.core.disposition import Disposition  # noqa: E402
