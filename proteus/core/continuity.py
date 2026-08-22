"""Harness-neutral continuity for context-fresh evolution phases.

Proteus deliberately starts a fresh model context for each phase in adapters that opt
into framework continuity.  This module carries only an operational handoff across that
boundary.  It lives outside the measured harness snapshot: a framework-generated note
must never be mistaken for self-evolution.

Adapters own native log parsing and phase execution.  The framework owns the protocol,
storage, redaction, history, and the prompt contract.  A handoff written explicitly by
the agent wins; when a phase is interrupted before it writes one, a deterministic summary
of normalized tool calls is used.  Raw model reasoning and tool results are never copied.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from proteus.core.adapter import ActionEvent

PROTOCOL_VERSION = 1
MODES = frozenset({"native", "framework", "none"})
# Containerized coding harnesses commonly restrict writes to /workspace.  The host source
# is still `<run>/harness`, while adapters bind this external directory over the nested
# container path below, so it is writable to the agent without entering the snapshot.
CONTAINER_ROOT = "/workspace/.proteus"
CONTAINER_HANDOFF = f"{CONTAINER_ROOT}/handoff.md"
MAX_CONTENT_CHARS = 12_000
MAX_PRIOR_CHARS = 6_000

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    ),
)


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(
            f"continuity_mode must be one of {sorted(MODES)}, got {mode!r}"
        )
    return mode


def framework_prompt(phase: str, *, goal_present: bool = True) -> str:
    """The portable protocol text appended to a framework-continuity phase prompt."""
    observe_action = (
        "Record objective-relevant findings and evidence for propose."
        if goal_present else
        "Record findings, evidence, and uncertainties for propose."
    )
    action = {
        "observe": observe_action,
        "propose": "Replace it with one scoped file-and-test plan for act.",
        "act": "Replace it with edits attempted, files changed, and verification still needed.",
        "reflect": "Replace it with validation results, unresolved risks, and the next step.",
    }.get(phase, "Replace it with concise continuation notes for the next phase.")
    return (
        f"Proteus continuity protocol v{PROTOCOL_VERSION}: this phase has a fresh model "
        f"context. Read {CONTAINER_HANDOFF} before acting; it is an external nested mount "
        f"and contains the prior phase "
        f"or episode handoff. Before ending, replace that file with a concise operational "
        f"handoff using these headings: Findings, Decision, Files, Tests, Open questions, "
        f"Next action. {action} Do not put credentials or raw tool output in the handoff."
    )


def _redact(text: str) -> str:
    out = text
    out = _SECRET_PATTERNS[0].sub("[REDACTED]", out)
    out = _SECRET_PATTERNS[1].sub("Bearer [REDACTED]", out)
    out = _SECRET_PATTERNS[2].sub(r"\1\2[REDACTED]", out)
    return out


def _clip(text: str, limit: int = MAX_CONTENT_CHARS) -> str:
    text = _redact(text).strip()
    if len(text) <= limit:
        return text
    return "[earlier handoff content omitted]\n\n" + text[-limit:]


def _event_detail(event: ActionEvent) -> str:
    preferred = ("file_path", "path", "pattern", "query", "command")
    detail = next((str(event.params[k]) for k in preferred if event.params.get(k)), "")
    detail = _redact(detail.replace("\n", " "))[:300]
    suffix = f": {detail}" if detail else ""
    surface = f" [{event.surface}]" if event.surface else ""
    return f"- {event.tool or 'action'}{surface}{suffix}"


def fallback_handoff(previous: str, events: Sequence[ActionEvent], interrupted: bool) -> str:
    """Build a provider-neutral fallback without model reasoning or tool results."""
    calls = [_event_detail(event) for event in events if event.tool]
    prior = _clip(previous, MAX_PRIOR_CHARS) if previous.strip() else "(none)"
    status = "interrupted before an explicit handoff" if interrupted else "no explicit handoff"
    activity = "\n".join(calls[-30:]) or "(no normalized tool calls recorded)"
    return _clip(
        "# Proteus operational handoff\n\n"
        f"Status: {status}. This fallback contains actions only, not hidden reasoning.\n\n"
        f"## Prior context\n\n{prior}\n\n"
        f"## Phase activity\n\n{activity}\n\n"
        "## Next action\n\nRe-check the current harness state, then continue from the evidence above."
    )


@dataclass(frozen=True)
class HandoffStart:
    episode: int
    phase: str
    previous: str
    baseline_sha256: str


class HandoffStore:
    """Run-local framework continuity store, always outside ``root/harness``."""

    def __init__(self, run_root: Path):
        self.run_root = Path(run_root)
        self.root = self.run_root / ".proteus-state"
        self.history = self.root / "handoffs"
        self.current = self.root / "handoff.md"
        self.latest = self.root / "latest.md"

    def initialise(self) -> None:
        self.history.mkdir(parents=True, exist_ok=True)
        meta = self.root / "continuity.json"
        if not meta.exists():
            self._atomic_text(meta, json.dumps({
                "protocol": "proteus-phase-continuity",
                "version": PROTOCOL_VERSION,
                "mode": "framework",
                "container_handoff": CONTAINER_HANDOFF,
                "persists_raw_reasoning": False,
            }, indent=2) + "\n")

    def begin(self, episode: int, phase: str) -> HandoffStart:
        """Expose the latest archived handoff and return a modification baseline."""
        self.initialise()
        previous = self.latest.read_text(encoding="utf-8") if self.latest.exists() else ""
        if not previous:
            previous = (
                "# Proteus operational handoff\n\n"
                "No prior phase has run. Inspect the current harness and write the first "
                "handoff before this phase ends.\n"
            )
        previous = _clip(previous)
        self._atomic_text(self.current, previous + ("\n" if previous else ""))
        digest = hashlib.sha256(self.current.read_bytes()).hexdigest()
        return HandoffStart(episode, phase, previous, digest)

    def reconcile(self, completed_episode: int) -> None:
        """Point live continuity at the last durable episode after crash recovery.

        Phase attempts beyond the snapshot checkpoint stay archived for diagnosis but
        must not feed a retried episode. Older runs may have no framework history; in that
        case the resumed phase starts with an empty operational handoff.
        """
        self.initialise()
        chosen: Path | None = None
        if completed_episode > 0:
            phase_dir = self.history / f"ep{completed_episode:03d}"
            for phase in ("reflect", "act", "propose", "observe"):
                candidates = sorted(phase_dir.glob(f"{phase}*.md"))
                if candidates:
                    chosen = candidates[-1]
                    break
        if chosen is None:
            self.latest.unlink(missing_ok=True)
            self.current.unlink(missing_ok=True)
            return
        content = _clip(chosen.read_text(encoding="utf-8")) + "\n"
        self._atomic_text(self.latest, content)
        self._atomic_text(self.current, content)

    def finish(self, start: HandoffStart, events: Sequence[ActionEvent] = (),
               interrupted: bool = False) -> dict:
        """Archive one phase and make its handoff available to the next fresh context."""
        self.initialise()
        try:
            current = self.current.read_text(encoding="utf-8")
            digest = hashlib.sha256(self.current.read_bytes()).hexdigest()
        except OSError:
            current, digest = "", ""
        explicit = bool(current.strip()) and digest != start.baseline_sha256
        content = _clip(current) if explicit else fallback_handoff(
            start.previous, events, interrupted
        )
        calls = [_event_detail(event) for event in events if event.tool][-30:]
        phase_dir = self.history / f"ep{start.episode:03d}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        stem, attempt = start.phase, 1
        while (phase_dir / f"{stem}.json").exists() or (phase_dir / f"{stem}.md").exists():
            attempt += 1
            stem = f"{start.phase}-{attempt:02d}"
        record = {
            "protocol_version": PROTOCOL_VERSION,
            "episode": start.episode,
            "phase": start.phase,
            "attempt": attempt,
            "source": "agent" if explicit else "framework-fallback",
            "interrupted": bool(interrupted),
            "content": content,
            "tool_calls": calls,
        }
        self._atomic_text(phase_dir / f"{stem}.json",
                          json.dumps(record, indent=2) + "\n")
        self._atomic_text(phase_dir / f"{stem}.md", content + "\n")
        self._atomic_text(self.latest, content + "\n")
        self._atomic_text(self.current, content + "\n")
        return record

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
