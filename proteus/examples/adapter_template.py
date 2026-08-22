"""Template: a new harness adapter, ready to copy.

Copy this file to `proteus/adapters/<yourname>.py` (or scaffold it with
`python -m proteus.scaffold adapter <YourName>`), then work through the TODOs. When it
passes

    proteus check --harness proteus.examples.adapter_template:TemplateHarness --episode

your harness satisfies the whole `HarnessAdapter` contract and Proteus can evolve,
measure, and swap dispositions in and out of it. Register a short name for it in
`proteus/cli.py::_adapter_factory` if you want `--harness <shortname>` instead of the
`<module>:<Class>` form.

This template is intentionally offline and dependency-free (no model, no Docker) so it
runs in tests and demos exactly like `proteus.adapters.minimal.MinimalHarness`, on which
it is modelled. A real harness replaces `run_episode` with a call into its own agent loop
(often inside a container — see `proteus.adapters.dsh`/`pi` for the from-source pattern),
but every other method usually stays this small.

The contract, in one breath: declare your editable **surfaces**, run **one episode** and
emit a normalized **trace**, and **install/remove a disposition** so divergence and
crystallization are measurable. Everything else (which model, how the loop is written) is
your harness's business. See `proteus/core/adapter.py` for the authoritative types.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

from proteus.core.adapter import ActionEvent, EpisodeResult, EpisodeSpec, Surface
from proteus.core.disposition import Disposition
from proteus.core.episode import PHASES  # ("observe", "propose", "act", "reflect")


class TemplateHarness:
    """A `HarnessAdapter` skeleton. Rename the class and `name`, then fill the TODOs."""

    #: Short, stable identifier for this harness. Appears in run manifests and reports.
    name = "template"

    #: How fresh phases continue across an episode. "none" (phases independent, like this
    #: stub and `minimal`), "native" (your harness owns continuity — the default if you
    #: omit this attribute entirely), or "framework" (let Proteus carry an external
    #: handoff note between phases; framework adapters mount `.proteus-state` in the
    #: container — see `proteus/core/continuity.py`). `proteus check` rejects any other value.
    continuity_mode = "none"

    #: Optional, for a harness whose OWN editable files can change how it executes (a
    #: self-editing coding agent — the highest-value target). Declare `staged_activation
    #: = True` and Proteus runs every phase from a frozen snapshot (`EpisodeSpec.active_root`)
    #: while `root/harness` stays the writable candidate, so an edit only takes effect the
    #: *next* episode and can be gated by an optional `validate_candidate(harness_root) -> str`.
    #: Leave it off for a harness that does not execute its own edited files.
    # staged_activation = True
    # def validate_candidate(self, harness_root: Path) -> str: ...

    #: True only if `install_disposition` writes the perturbation into a file the harness
    #: reads on its own (an `AGENTS.md`-style instructions surface). Leave False and the
    #: framework appends the disposition text to every phase prompt instead. Declaring
    #: True when you also carry it in a file would double-dose the perturbation — read the
    #: note on `HarnessAdapter.disposition_in_files` before flipping this.
    disposition_in_files = False

    # --- static declaration -------------------------------------------------------------
    def surfaces(self) -> Sequence[Surface]:
        """The persistent, agent-editable regions the measurement layer counts over.

        TODO: declare one Surface per region your harness lets the agent edit. `unit` is
        "file" | "directory" | "top_level_def"; set `is_code=True` for regions that must
        still run after an edit (they pass the viability gate); `write_tools` names the
        tool calls that count as edits to this surface.
        """
        return [
            Surface("notes", "notes", unit="file",
                    write_tools=frozenset({"write_note"})),
        ]

    def required_edit_tools(self) -> frozenset[str]:
        """Tools whose presence proves the harness can still edit itself (viability)."""
        return frozenset({"write_note"})

    # --- provisioning -------------------------------------------------------------------
    def seed(self, harness_root: Path, rng_seed: int = 0) -> None:
        """Materialise a fresh copy of the harness at `harness_root` (episode-0 state).

        TODO: lay down every surface directory and any baseline files the harness needs
        to boot. For a from-source harness this is where you extract the real source from
        the prepared image (see `dsh`/`pi`). `rng_seed` is the replicate index, for
        harnesses that need deterministic per-seed naming at provisioning time.
        """
        (harness_root / "notes").mkdir(parents=True, exist_ok=True)
        (harness_root / "STATE.md").write_text("# template harness\nepisode 0\n")

    def install_disposition(self, harness_root: Path, disposition: Disposition) -> None:
        """Apply the action-preference perturbation. MUST be removable.

        Reinstalling `NEUTRAL` must restore the exact fingerprint you had before any
        disposition was installed — `proteus check` enforces this, because
        crystallization mounts the pre-perturbation state (F0). Here we serialise the
        disposition to a dotfile; a file-carrying harness would instead write it into its
        own instructions surface and set `disposition_in_files = True`.
        """
        (harness_root / ".disposition.json").write_text(json.dumps({
            "label": disposition.label,
            "config": dict(disposition.config),
            "empty": disposition.is_empty,
        }))

    # --- run + observe ------------------------------------------------------------------
    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult:
        """Run exactly one context-fresh episode against the harness at `spec.root`.

        TODO: replace the deterministic stub below with a call into your agent loop. The
        real work is: give the loop `spec.phase_prompts` (the default episode protocol,
        goal, and visible evaluator feedback are already merged in), let it edit the
        surfaces, and write a per-turn trace that `read_trace` can reload. Results flow
        through the trace, never through stdout.

        The writable harness is at `spec.root / "harness"`. If you declared
        `staged_activation = True`, execute from `spec.active_root` instead and keep the
        candidate separate. Turn budget: `spec.max_turns` bounds the episode;
        `spec.min_turns_per_phase` reserves turns for later phases (a phase that reaches
        its reserved line ends early — only a spent budget ends the episode). Honour it or
        the mid-phase cap is untestable.
        """
        harness = spec.root / "harness"
        (harness / "notes").mkdir(parents=True, exist_ok=True)  # restore may drop empty dirs
        rng = random.Random(f"{spec.seed}:{spec.root.name}:{spec.episode}")
        trace_path = spec.root / "traces" / f"ep{spec.episode:03d}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        turn = 0
        writes = 0
        capped = False
        min_pp = int(getattr(spec, "min_turns_per_phase", 0) or 0)
        with trace_path.open("w", encoding="utf-8") as sink:
            for idx, phase in enumerate(PHASES):
                if capped:
                    break
                stop_at = (spec.max_turns - min_pp * (len(PHASES) - idx - 1)
                           if spec.max_turns else 0)
                prompt = spec.phase_prompts.get(phase, "")
                for tool, surface, text in self._stub_policy(phase, prompt, spec.episode, rng):
                    if spec.max_turns and turn >= stop_at:
                        capped = turn >= spec.max_turns
                        break
                    turn += 1
                    if tool == "write_note":
                        (harness / "notes" / f"{text}.md").write_text(
                            f"episode {spec.episode}: {text}\n")
                        writes += 1
                    sink.write(json.dumps({
                        "turn": turn, "phase": phase, "tool": tool,
                        "surface": surface, "text": text,
                    }) + "\n")
        return EpisodeResult(episode=spec.episode, ok=True, turns=turn,
                             counters={"writes": writes, "turn_capped": capped})

    @staticmethod
    def _stub_policy(phase: str, prompt: str, episode: int, rng: random.Random):
        """Deterministic offline stand-in for a real agent loop. Delete when you wire yours."""
        if phase == "observe":
            return [("read_state", None, "surveyed the harness")]
        if phase == "act":
            return [("write_note", "notes", f"note_e{episode}_{rng.randint(0, 3)}")]
        return []

    def read_trace(self, root: Path, episode: int) -> Sequence[ActionEvent]:
        """Reload one episode's trace as normalized `ActionEvent`s.

        TODO: if your loop writes its own log format, translate it here. Phases must be
        the canonical four (observe/propose/act/reflect); `tool=None` marks a text-only
        turn; `surface` names the surface an edit targeted, when applicable.
        """
        path = root / "traces" / f"ep{episode:03d}.jsonl"
        out: list[ActionEvent] = []
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(ActionEvent(turn=d["turn"], phase=d["phase"], tool=d.get("tool"),
                                   surface=d.get("surface"), text=d.get("text", "")))
        return out

    # --- immutability observable --------------------------------------------------------
    def disposition_fingerprint(self, harness_root: Path) -> str:
        """A hash of the currently-installed disposition, to detect drift of F over time.

        Must return the identical value before install and after removal (see
        `install_disposition`).
        """
        p = harness_root / ".disposition.json"
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


if __name__ == "__main__":
    # Smoke it exactly the way `proteus check` does, but inline and offline.
    from proteus.testing import check_adapter

    raise SystemExit(len(check_adapter(TemplateHarness(), episode=True)))
