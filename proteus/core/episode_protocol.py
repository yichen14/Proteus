"""The versioned default episode protocol.

Proteus supplies a reference observe/propose/act/reflect protocol while adapters remain
free to decide how those phases execute.  The protocol keeps goals and evaluators
independent: external evaluation can be a complete operationalization of a narrow goal or
only partial evidence for a broad one, and the evolving harness decides whether additional
self-owned tests or evaluators would reduce uncertainty.

No-goal runs use neutral exploration language.  They are not silently turned into
"improve reliability" runs, and the harness remains free to formulate its own provisional
goals and evaluators as part of its evolving state.
"""

from __future__ import annotations

from typing import Mapping


DEFAULT_EPISODE_PROTOCOL_VERSION = 1


GOAL_PHASE_PROMPTS: Mapping[str, str] = {
    "observe": (
        "Take stock of the harness you woke up in: what is here, what state it is in, "
        "and what evidence is relevant to the objective."
    ),
    "propose": (
        "Choose one scoped improvement to pursue next and form an actionable file-and-test "
        "plan."
    ),
    "act": "Carry out the scoped plan by editing your own harness.",
    "reflect": (
        "Validate what changed, identify unresolved risks, and choose the next concrete "
        "step."
    ),
}


OPEN_PHASE_PROMPTS: Mapping[str, str] = {
    "observe": (
        "Take stock of the harness you woke up in: what is here, what state it is in, "
        "what has changed, and what evidence or uncertainty seems worth examining. Do "
        "not assume an unstated objective."
    ),
    "propose": (
        "Choose one scoped experiment, question, or change you judge worth pursuing next "
        "and form an actionable plan, including what you expect to observe."
    ),
    "act": "Carry out the scoped plan by editing or probing your own harness.",
    "reflect": (
        "Compare what happened with what you expected; record effects, surprises, "
        "tradeoffs, and unresolved questions; and choose the next concrete step."
    ),
}


EPISTEMIC_PROTOCOL = f"""\
Proteus epistemic protocol v{DEFAULT_EPISODE_PROTOCOL_VERSION}: If external evaluators or
evaluator feedback are supplied, treat them as evidence about your evolution, not
automatically as a complete definition of success. An evaluator may fully operationalize
the stated goal—for example, when the goal is specifically to improve that benchmark—or
it may cover only part of a broader natural-language objective. Judge its sufficiency
against the actual goal when one exists. Add, revise, or retain harness-owned tests and
evaluators when doing so would materially reduce uncertainty about a goal you are pursuing;
do not add them merely to satisfy this protocol.

If no external goal is supplied, do not assume one. You may continue open-ended
exploration, formulate or revise your own provisional goals, and create your own evaluators
when you judge them useful. Any such goal or evaluator becomes part of the harness's
evolving state, not an objective supplied by Proteus.
"""


def default_phase_prompts(goal_text: str) -> dict[str, str]:
    """Return fresh mutable prompts for the stated-goal or open-ended default."""
    source = GOAL_PHASE_PROMPTS if goal_text.strip() else OPEN_PHASE_PROMPTS
    return dict(source)
