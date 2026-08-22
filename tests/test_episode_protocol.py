"""The default episode protocol stays goal/evaluator independent."""

from pathlib import Path

from proteus.adapters.minimal import MinimalHarness
from proteus.core import EvaluatorSpec, GoalConfig, NEUTRAL, Visibility
from proteus.core.episode import RunConfig, _phase_prompts
from proteus.core.episode_protocol import (
    DEFAULT_EPISODE_PROTOCOL_VERSION,
    EPISTEMIC_PROTOCOL,
)
from proteus.core.goal import EvalResult


def _prompts(goal: GoalConfig, prior_feedback: str = "") -> dict[str, str]:
    cfg = RunConfig(
        name="protocol", adapter=MinimalHarness(), disposition=NEUTRAL, goal=goal,
        root=Path("/nonexistent"), model="mock", episodes=1,
    )
    return _phase_prompts(cfg, prior_feedback)


def _score(trace, ctx):
    del trace, ctx
    return EvalResult(name="external", score=1.0, passed=True)


def test_protocol_leaves_evaluator_sufficiency_to_the_harness():
    normalized = " ".join(EPISTEMIC_PROTOCOL.split())
    assert f"epistemic protocol v{DEFAULT_EPISODE_PROTOCOL_VERSION}" in normalized
    assert "may fully operationalize the stated goal" in normalized
    assert "it may cover only part" in normalized
    assert "Judge its sufficiency against the actual goal" in normalized
    assert "do not add them merely to satisfy this protocol" in normalized

    prompts = _prompts(GoalConfig.of(
        text="Improve benchmark X.",
        evaluators=(EvaluatorSpec(
            name="benchmark-x", run=_score, kind="benchmark",
            visibility=Visibility.OBSERVE,
        ),),
    ))
    assert all(prompt.count("Proteus epistemic protocol") == 1
               for prompt in prompts.values())
    assert all("Improve benchmark X." in prompt for prompt in prompts.values())


def test_no_goal_default_is_open_ended_and_may_evolve_its_own_goal():
    prompts = _prompts(GoalConfig.no_goal())
    assert all("If no external goal is supplied, do not assume one" in prompt
               for prompt in prompts.values())
    assert all("formulate or revise your own provisional goals" in prompt
               for prompt in prompts.values())
    assert "Do not assume an unstated objective" in prompts["observe"]
    assert "scoped experiment, question, or change" in prompts["propose"]
    assert "editing or probing" in prompts["act"]
    assert "effects, surprises" in prompts["reflect"]
    assert all("Evolution objective for this run" not in prompt
               for prompt in prompts.values())
    assert all("scoped improvement" not in prompt for prompt in prompts.values())


def test_goal_without_evaluator_still_receives_the_same_protocol():
    prompts = _prompts(GoalConfig.of(text="Become more reliable."))
    assert all("Become more reliable." in prompt for prompt in prompts.values())
    assert "evidence is relevant to the objective" in prompts["observe"]
    assert "scoped improvement" in prompts["propose"]
    assert all("harness-owned tests and evaluators" in " ".join(prompt.split())
               for prompt in prompts.values())


def test_protocol_does_not_reveal_hidden_evaluator_identity():
    goal = GoalConfig.of(evaluators=(EvaluatorSpec(
        name="private-evaluator-name", run=_score, visibility=Visibility.HIDDEN,
    ),))
    prompts = _prompts(goal)
    assert all("private-evaluator-name" not in prompt for prompt in prompts.values())
    assert all("Feedback on your last episode" not in prompt for prompt in prompts.values())


def test_no_goal_framework_handoff_uses_neutral_observe_language(tmp_path):
    class FreshAdapter:
        continuity_mode = "framework"
        disposition_in_files = False

    cfg = RunConfig(
        name="protocol", adapter=FreshAdapter(), disposition=NEUTRAL,
        goal=GoalConfig.no_goal(), root=tmp_path, model="mock", episodes=1,
    )
    observe = _phase_prompts(cfg, "")["observe"]
    assert "Record findings, evidence, and uncertainties for propose" in observe
    assert "objective-relevant findings" not in observe


def test_shipped_no_goal_carriers_do_not_impose_improvement_language():
    from proteus.adapters.dsh import SEED_INSTRUCTIONS as DSH_INSTRUCTIONS
    from proteus.adapters.llm import SYSTEM as LLM_SYSTEM
    from proteus.adapters.pi import SEED_INSTRUCTIONS as PI_INSTRUCTIONS

    for text in (DSH_INSTRUCTIONS, PI_INSTRUCTIONS, LLM_SYSTEM):
        assert "maintain and improve" not in text.lower()
