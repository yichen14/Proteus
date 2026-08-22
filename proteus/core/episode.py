"""The self-evolution loop: run a harness for N context-fresh episodes and record its
trajectory.

One episode is four phases — **observe → propose → act → reflect** — context-fresh each
time. Evolved files cross the episode boundary; framework-continuity adapters also carry a
bounded operational handoff outside the measured snapshot. The framework owns everything
that is *not* the harness: it builds each phase's prompt (folding in the versioned default
episode protocol, goal text, and any evaluator feedback the agent is allowed to see),
defines the continuity protocol, asks the adapter to run the episode, snapshots the working
tree, runs the evaluators, and applies the outer-loop selection if one is configured. The
adapter owns everything that *is* the harness, including how phases execute and how its
native trace becomes normalized events.

This separation is what makes Proteus harness-agnostic and condition-complete at once: the
same framework runs Aki or a bare ReAct loop, under no-goal or multi-goal, with evaluators
hidden or visible, and the measurement layer reads all of it with one ruler.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from proteus.core import snapshot
from proteus.core.adapter import EpisodeSpec, HarnessAdapter
from proteus.core.budget import PHASES, make_budget_plan
from proteus.core.disposition import Disposition
from proteus.core.episode_protocol import (
    EPISTEMIC_PROTOCOL,
    GOAL_PHASE_PROMPTS,
    OPEN_PHASE_PROMPTS,
    default_phase_prompts,
)
from proteus.core.goal import GoalConfig, GoalContext

def _write_json_atomic(path: Path, value) -> None:
    """Replace one JSON record without exposing a truncated crash-time file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1), encoding="utf-8")
    temporary.replace(path)


def private_record_dir(root: Path) -> Path:
    """Framework-owned records outside the subject-visible run root."""
    root = Path(root)
    return root.parent / ".proteus-records" / root.name


def eval_history_path(root: Path) -> Path:
    """Durable evaluator history, including scores hidden from the subject."""
    return private_record_dir(root) / "eval_history.json"


PENDING_CANDIDATE_VERSION = 1


def pending_candidate_path(root: Path) -> Path:
    """Framework-private pointer to a failed staged candidate awaiting repair."""
    return private_record_dir(root) / "pending_candidate.json"


def _write_pending_candidate(root: Path, *, commit: str, resume_episode: int,
                             reason: str, error: str) -> dict:
    record = {
        "version": PENDING_CANDIDATE_VERSION,
        "commit": commit,
        "resume_episode": resume_episode,
        "reason": reason,
        "error": str(error)[:2000],
    }
    _write_json_atomic(pending_candidate_path(root), record)
    return record


def _load_pending_candidate(root: Path, harness: Path, expected_episode: int) -> dict | None:
    """Load the exact failed tree to use as a staged episode's writable repair base."""
    path = pending_candidate_path(root)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        version = int(record["version"])
        commit = str(record["commit"])
        resume_episode = int(record["resume_episode"])
        reason = str(record["reason"])
        error = str(record.get("error", ""))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pending staged candidate is unreadable at {path}: {exc}") from exc
    if version != PENDING_CANDIDATE_VERSION:
        raise ValueError(
            f"pending staged candidate at {path} has unsupported version {version}"
        )
    if resume_episode < expected_episode:
        # The process can die after committing an accepted episode but before deleting
        # the repair pointer. The durable episode checkpoint wins; this pointer is stale.
        path.unlink(missing_ok=True)
        return None
    if resume_episode != expected_episode:
        raise ValueError(
            f"pending staged candidate targets episode {resume_episode}, but the next "
            f"durable episode is {expected_episode}"
        )
    if not snapshot.has_commit(harness, commit):
        raise ValueError(
            f"pending staged candidate {commit!r} is missing from the snapshot repository"
        )
    return {
        "version": version,
        "commit": commit,
        "resume_episode": resume_episode,
        "reason": reason,
        "error": error,
    }


def _clear_pending_candidate(root: Path) -> None:
    pending_candidate_path(root).unlink(missing_ok=True)


def _repair_feedback(pending: dict, prior: str = "") -> str:
    detail = str(pending.get("error", ""))[:600]
    notice = (
        "Proteus restored the exact failed candidate as this episode's writable repair "
        "base. The running harness is still the last valid snapshot; inspect and fix the "
        "candidate rather than rebuilding the change from scratch."
    )
    if detail:
        notice += f" Previous failure: {detail}"
    return f"{notice}\n\n{prior}" if prior else notice


# Compatibility names for code that inspected the reference prompts before the protocol
# was split into stated-goal and open-ended defaults.
BASE_PROMPTS: Mapping[str, str] = GOAL_PHASE_PROMPTS
OPEN_BASE_PROMPTS: Mapping[str, str] = OPEN_PHASE_PROMPTS


@dataclass
class RunConfig:
    name: str
    adapter: HarnessAdapter
    disposition: Disposition
    goal: GoalConfig
    root: Path
    model: str
    episodes: int = 30
    max_turns: int = 100
    min_turns_per_phase: int = 0
    """Per-phase floor on the turn budget (see EpisodeSpec). `max_turns` must be at
    least `len(PHASES) * min_turns_per_phase`."""
    phase_turns: Mapping[str, int] = field(default_factory=dict)
    """Explicit normal allocation for all four phases; must sum to ``max_turns``."""
    hard_max_turns: int = 0
    """Burst ceiling for an explicit phase plan. Zero uses ``max_turns``."""
    checkpoint_turns: int = 0
    """End-of-phase calls reserved for the harness's persistent handoff mechanism."""
    seed: int = 0
    task: object | None = None
    """A `proteus.bench.BenchTask` to seed before episode 1. Its workspace is
    `<run>/task/`, beside the measured `<run>/harness/` and outside the snapshot. An
    adapter that supports benchmark work must expose that sibling to its agent; dsh/pi
    mount it at `/workspace/task`."""
    grader_sandbox: object | None = None
    """Optional isolated runner for agent-authored benchmark code. Local/polyglot use a
    networkless Docker grader by default; host execution is never a fallback."""
    announce_budget: bool = False
    """Tell the agent its live phase allocation and episode limits, so it can plan within
    them. Off by default: announcing a budget changes behaviour — that is the point — so
    it is an experimental condition, recorded in the manifest, not a silent default.
    Enforcement is separate and always on where possible: direct stops for in-process
    harnesses, between-phase checks and mid-phase log watching for containerized ones."""
    progress_path: Path | None = None
    """Where to append one JSON line per finished episode (live tracking). Must live
    OUTSIDE `root`: the subject agent can read its own run root, and a progress record
    carries the condition label and HIDDEN evaluator scores."""


@dataclass
class RunResult:
    name: str
    episodes_complete: int
    root: str
    error: str = ""
    eval_history: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    """Numeric adapter counters summed across episodes (tokens_in/tokens_out where the
    adapter reports them) — what a cost estimate is built from."""


def _phase_prompts(cfg: RunConfig, prior_feedback: str) -> dict[str, str]:
    """Assemble one episode's four default protocol prompts.

    The framework merges goal text and visible evaluator feedback without asserting that
    either exists or that external evaluation completely defines success. The agent never
    sees anything about why a condition was configured.
    """
    gt = cfg.goal.goal_text()
    has_goal = bool(gt.strip())
    prompts = default_phase_prompts(gt)
    from proteus.core.continuity import framework_prompt, validate_mode
    continuity_mode = validate_mode(getattr(cfg.adapter, "continuity_mode", "native"))
    if continuity_mode == "framework":
        for ph in PHASES:
            prompts[ph] = f"{prompts[ph]}\n\n{framework_prompt(ph, goal_present=has_goal)}"
    if getattr(cfg.adapter, "staged_activation", False):
        staging_note = (
            "Episode isolation contract: the harness running this phase is the frozen "
            "last-valid snapshot at /workspace. Your writable candidate is mounted at "
            "/workspace/candidate. Read the active harness to understand current behavior, "
            "but make every persistent edit under /workspace/candidate. Candidate changes "
            "do not become the running harness in any phase of this episode, including "
            "reflect; do not replace or reload the active process from the candidate. "
            "Reflect may inspect the candidate and its diff. Proteus validates it after "
            "reflect and activates it only in the next episode if the gate passes. If a "
            "prior candidate failed viability, this writable tree is that exact failed "
            "candidate restored for repair; the active runtime still remains last-valid."
        )
        for ph in PHASES:
            prompts[ph] = f"{staging_note}\n\n{prompts[ph]}"
    # This is part of Proteus's default episode protocol, not a disposition and not an
    # evaluator. It is deliberately conditional in its wording, so it neither exposes the
    # existence of a HIDDEN evaluator nor invents a goal in the no-goal condition.
    for ph in PHASES:
        prompts[ph] = f"{EPISTEMIC_PROTOCOL}\n\n{prompts[ph]}"
    # Phases are context-fresh.  Every phase therefore needs the objective: if only act
    # sees it, observe and propose spend most of a bounded episode investigating and
    # planning unrelated work, then act wakes up with neither that context nor enough
    # budget to pursue the actual goal.  Empty text preserves the no-goal condition.
    if has_goal:
        objective = f"Evolution objective for this run:\n{gt}"
        for ph in PHASES:
            prompts[ph] = f"{objective}\n\n{prompts[ph]}"
    # evaluator feedback the agent is allowed to see enters the observe phase
    if prior_feedback:
        prompts["observe"] = f"{prior_feedback}\n\n{prompts['observe']}"
    # the budget announcement comes first: it frames how the agent plans the episode
    if cfg.announce_budget and cfg.max_turns:
        hard = cfg.hard_max_turns or cfg.max_turns
        note = (f"Budget condition: you have at most {hard} tool calls in this episode; "
                f"the normal plan is {cfg.max_turns} across all phases. The adapter will "
                "report live used and remaining counts at phase start.")
        if cfg.phase_turns:
            allocation = ", ".join(
                f"{phase}={cfg.phase_turns[phase]}" for phase in PHASES
            )
            note += f" Planned phase allocation: {allocation}; unused quota goes to act."
        elif cfg.min_turns_per_phase:
            note += (f" Each later phase reserves at least {cfg.min_turns_per_phase} "
                     "calls; a phase may be ended early to protect that reserve.")
        for ph in PHASES:
            prompts[ph] = f"{note}\n\n{prompts[ph]}"
    # the disposition contributes its (per-phase) text — unless the adapter already carries
    # it in a file the harness loads itself, in which case adding it here would deliver the
    # same perturbation twice per phase (see HarnessAdapter.disposition_in_files)
    if not getattr(cfg.adapter, "disposition_in_files", False):
        for ph in PHASES:
            suffix = cfg.disposition.phase_text(ph)
            if suffix:
                prompts[ph] = f"{prompts[ph]}\n\n{suffix}"
    return prompts


def _append_progress(cfg: RunConfig, ep: int, res, trace, accepted: bool, results) -> None:
    """One JSON line per finished episode, for the live report. Never inside cfg.root."""
    import time

    from proteus.measure import distance
    units = distance.units(cfg.root / "harness", cfg.adapter.surfaces())
    rec = {
        "ts": time.time(), "name": cfg.name, "seed": cfg.seed, "episode": ep,
        "episodes_target": cfg.episodes, "ok": res.ok, "turns": res.turns,
        "error": res.error,
        "tool_calls": sum(1 for e in trace if e.tool),
        "units": {k: len(v) for k, v in units.items()},
        "accepted": accepted,
        "scores": {r.name: r.score for r in results},
        "counters": dict(res.counters or {}),
    }
    cfg.progress_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.progress_path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(rec) + "\n")


def completed_episodes(cfg: RunConfig) -> int:
    """Contiguous episodes already snapshotted under `cfg.root`, for mid-seed resume.

    Counts commits, not trace files: a provider outage writes a trace per failed attempt,
    and counting those reports a seed that never finished an episode as complete. Counting
    is contiguous from 1 because a gap means the chain the measurement walks is broken.
    """
    harness = cfg.root / "harness"
    if not (harness.parent / ".snapshot.git").exists():
        return 0
    ep = 0
    while snapshot.commit_for_episode(harness, ep + 1) is not None:
        ep += 1
    return ep


def run(cfg: RunConfig, start: int = 0, *, resume: bool = False) -> RunResult:
    """Run one seed's full trajectory, harness retained under `cfg.root`.

    `start` resumes an interrupted seed: episodes up to and including `start` are taken as
    done and the harness on disk is used as-is. `resume=True` is required to distinguish
    recovery from episode 0 from a genuinely fresh run — both have `start == 0`, but only
    the former already has a snapshot that must be restored before retrying episode 1.
    Any positive `start` implies resume for backward compatibility. Re-seeding a resumed
    root would overwrite the evolved harness with fresh templates, so provisioning is
    skipped entirely.
    """
    harness = cfg.root / "harness"
    records = private_record_dir(cfg.root)
    staged_activation = bool(getattr(cfg.adapter, "staged_activation", False))
    make_budget_plan(
        max_turns=cfg.max_turns,
        min_turns_per_phase=cfg.min_turns_per_phase,
        phase_turns=cfg.phase_turns,
        hard_max_turns=cfg.hard_max_turns,
        checkpoint_turns=cfg.checkpoint_turns,
    )
    if cfg.checkpoint_turns and not cfg.announce_budget:
        raise ValueError("checkpoint_turns requires announce_budget")
    if cfg.checkpoint_turns and getattr(cfg.adapter, "continuity_mode", "native") == "none":
        raise ValueError(
            "checkpoint_turns requires a harness with native or framework continuity"
        )
    completed = completed_episodes(cfg)
    is_resume = resume or bool(start)
    if is_resume:
        if completed != start:
            raise ValueError(
                f"cannot resume {cfg.root} at episode {start}: snapshot history has "
                f"only {completed} completed episodes; resume must start at the exact durable "
                "checkpoint")
        checkpoint = snapshot.commit_for_episode(harness, start)
        if checkpoint is None:  # guarded by completed == start; keep the failure legible
            raise ValueError(
                f"cannot resume {cfg.root}: episode {start} checkpoint is missing")
        # A normal adapter/provider failure has already committed its staged candidate.
        # SIGKILL cannot run that handler, so capture any dirty (or just-committed) staged
        # tree before resetting. It is a repair base only: the executable active snapshot
        # still comes from `checkpoint` below.
        interrupted = ""
        if staged_activation:
            interrupted = snapshot.preserve_interrupted_candidate(
                harness, checkpoint, start + 1,
                f"candidate {start + 1}: {cfg.name} [interrupted before checkpoint]",
            )
        else:
            snapshot.reset_to_checkpoint(harness, checkpoint)
        if interrupted:
            _write_pending_candidate(
                cfg.root, commit=interrupted, resume_episode=start + 1,
                reason="interrupted", error="process stopped before an episode checkpoint",
            )
    else:
        # A fresh/overwrite run must not inherit hidden scores or an F baseline from an
        # older run directory with the same deterministic id.
        shutil.rmtree(records, ignore_errors=True)
        cfg.adapter.seed(harness, cfg.seed)
        cfg.adapter.install_disposition(harness, cfg.disposition)
        if cfg.task is not None:
            from proteus.bench.task import seed_task
            seed_task(harness, cfg.task)
        snapshot.init(harness)

    # Resume must restore the experiment's state, not just its files: the selection
    # baseline, the visible feedback, and the cumulative counters all live in
    # eval_history, and a resume that reset them would let accept_reject approve a
    # post-resume episode worse than everything before the interruption.
    eval_history: list[dict] = []
    prior_feedback = ""
    totals: dict = {}
    best_score: float | None = None
    history_path = eval_history_path(cfg.root)
    fingerprint_path = records / "disposition_fingerprint.json"
    fingerprint = cfg.adapter.disposition_fingerprint(harness)
    if not is_resume:
        _write_json_atomic(fingerprint_path, {"fingerprint": fingerprint})
    if is_resume:
        if start and not history_path.exists():
            raise ValueError(
                f"cannot resume {cfg.root}: {start} episodes are snapshotted but "
                f"private eval history is missing at {history_path}")
        if history_path.exists():
            try:
                eval_history = json.loads(history_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(
                    f"cannot resume {cfg.root}: private eval history is unreadable: {exc}"
                ) from exc
        expected = list(range(1, start + 1))
        recorded = [row.get("episode") for row in eval_history]
        if len(eval_history) != start or recorded != expected:
            raise ValueError(
                f"cannot resume {cfg.root}: snapshot history has {start} episodes but "
                f"eval history records {recorded}; refusing a desynchronised run")
        try:
            installed = json.loads(fingerprint_path.read_text(encoding="utf-8"))
            fingerprint = str(installed["fingerprint"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            if start:
                raise ValueError(
                    f"cannot resume {cfg.root}: disposition fingerprint record is unreadable: "
                    f"{exc}"
                ) from exc
            # A machine can die after episode-0 is committed but before this small private
            # record is replaced. The restored checkpoint is the only completed state, so
            # its fingerprint is a safe baseline from which to reconstruct the record.
            fingerprint = cfg.adapter.disposition_fingerprint(harness)
            _write_json_atomic(fingerprint_path, {"fingerprint": fingerprint})
        current = cfg.adapter.disposition_fingerprint(harness)
        checkpoint_fingerprint = (
            str(eval_history[-1].get("disposition_fingerprint", ""))
            if start else fingerprint
        )
        if not checkpoint_fingerprint or current != checkpoint_fingerprint:
            raise ValueError(
                f"cannot resume {cfg.root}: current disposition fingerprint {current!r} "
                f"does not match the last durable checkpoint {checkpoint_fingerprint!r}"
            )
        if getattr(cfg.adapter, "continuity_mode", "native") == "framework":
            from proteus.core.continuity import HandoffStore
            HandoffStore(cfg.root).reconcile(start)
        for row in eval_history:
            results = row.get("results") or []
            if results:
                score = sum(r.get("score", 0.0) for r in results) / len(results)
                if row.get("accepted") and (best_score is None or score >= best_score):
                    best_score = score
            for key, value in (row.get("counters") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] = totals.get(key, 0) + value
        if eval_history:
            last = eval_history[-1]
            from proteus.core.goal import EvalResult
            by_name = {r["name"]: EvalResult(**r) for r in (last.get("results") or [])}
            prior_feedback = cfg.goal.observe_feedback(by_name)
            if last.get("failure_kind") == "viability":
                recovery = (
                    "Your last candidate failed the episode-boundary viability gate and "
                    "was rolled back. Fix the underlying issue in a new candidate: "
                    f"{str(last.get('error', 'validation failed'))[:600]}"
                )
                prior_feedback = (f"{recovery}\n\n{prior_feedback}"
                                  if prior_feedback else recovery)
            elif prior_feedback and not last.get("accepted", True):
                prior_feedback += "\n(Your last episode's changes were not kept.)"
    pending = (
        _load_pending_candidate(cfg.root, harness, start + 1)
        if staged_activation else None
    )
    if pending:
        if pending["reason"] == "viability" and eval_history:
            # Replace the older generic rollback wording while retaining only evaluator
            # feedback the agent is allowed to observe.
            from proteus.core.goal import EvalResult
            last_results = {
                r["name"]: EvalResult(**r) for r in (eval_history[-1].get("results") or [])
            }
            prior_feedback = cfg.goal.observe_feedback(last_results)
        prior_feedback = _repair_feedback(pending, prior_feedback)
    error = ""
    done = start
    last_checkpoint = snapshot.head(harness)  # gapless episode mapping, including rollbacks
    last_accepted = last_checkpoint            # same valid tree at start/resume
    for ep in range(start + 1, cfg.episodes + 1):
        active_root = None
        if staged_activation:
            # Keep the executable snapshot outside both the writable candidate and the
            # agent-visible handoff mount. If it lived under `.proteus-state`, the latter's
            # writable nested mount would give the subject a back door into its supposedly
            # frozen runtime between phases.
            shutil.rmtree(cfg.root / ".proteus-state" / "active", ignore_errors=True)
            active_root = private_record_dir(cfg.root) / "active"
            shutil.rmtree(active_root, ignore_errors=True)
            snapshot.materialize(harness, last_accepted, active_root)
            if pending and pending["resume_episode"] == ep:
                # Only the writable side resumes from the failure. HEAD stays on the
                # durable rollback checkpoint, and active_root was already materialised
                # from the last valid tree, so failed code cannot execute in this episode.
                snapshot.restore(harness, pending["commit"])
        spec = EpisodeSpec(
            root=cfg.root, episode=ep, model=cfg.model,
            phase_prompts=_phase_prompts(cfg, prior_feedback),
            max_turns=cfg.max_turns, seed=cfg.seed,
            min_turns_per_phase=cfg.min_turns_per_phase,
            phase_turns=dict(cfg.phase_turns), hard_max_turns=cfg.hard_max_turns,
            checkpoint_turns=cfg.checkpoint_turns,
            announce_budget=cfg.announce_budget,
            continuity_mode=getattr(cfg.adapter, "continuity_mode", "native"),
            active_root=active_root,
        )
        try:
            res = cfg.adapter.run_episode(spec)
        except Exception as exc:  # noqa: BLE001 - a failed episode is a record, not a crash
            error = f"{type(exc).__name__}: {exc}"
            try:
                failed_commit = snapshot.preserve_failed_candidate(
                    harness, last_checkpoint, ep,
                    f"candidate {ep}: {cfg.name} [run failed: {type(exc).__name__}]",
                )
                if staged_activation:
                    pending = _write_pending_candidate(
                        cfg.root, commit=failed_commit, resume_episode=ep,
                        reason="run_failed", error=error,
                    )
            except Exception as restore_exc:  # noqa: BLE001
                error += f"; automatic restore failed: {restore_exc}"
            break
        if not res.ok:
            error = res.error
            try:
                failed_commit = snapshot.preserve_failed_candidate(
                    harness, last_checkpoint, ep,
                    f"candidate {ep}: {cfg.name} [run failed]",
                )
                if staged_activation:
                    pending = _write_pending_candidate(
                        cfg.root, commit=failed_commit, resume_episode=ep,
                        reason="run_failed", error=error,
                    )
            except Exception as restore_exc:  # noqa: BLE001
                error += f"; automatic restore failed: {restore_exc}"
            break

        candidate_fingerprint = cfg.adapter.disposition_fingerprint(harness)

        trace = cfg.adapter.read_trace(cfg.root, ep)
        # A candidate may be inspected during reflect, but it is never executed inside the
        # model-driven episode. Only this boundary gate may build/run it, without a model
        # session. A failed candidate is preserved, rolled back, and
        # counted as a completed (rejected) episode so the next episode can recover. The
        # gate precedes arbitrary evaluators so invalid candidate code is never launched by
        # benchmark/user evaluation either.
        viability_error = ""
        validator = getattr(cfg.adapter, "validate_candidate", None)
        if validator is not None:
            try:
                viability_error = str(validator(harness) or "")
            except Exception as exc:  # noqa: BLE001 - validation failure is a rejection
                viability_error = f"{type(exc).__name__}: {exc}"

        # Evaluate a viable candidate BEFORE snapshotting, so selection can still reject
        # it. An evaluator is user (or benchmark) code — a crash in it must not take the
        # whole trajectory down; a failed evaluator records a zero and the run continues.
        results = []
        if not viability_error:
            try:
                results = cfg.goal.evaluate(
                    trace, GoalContext(str(harness), ep, grader_sandbox=cfg.grader_sandbox)
                )
            except Exception as exc:  # noqa: BLE001
                from proteus.core.goal import EvalResult
                results = [EvalResult(name="evaluator-error", score=0.0,
                                      detail=f"{type(exc).__name__}: {exc}"[:200])]
        by_name = {r.name: r for r in results}

        # outer-loop selection on the scores (visibility-independent: an outer loop may
        # act on scores the agent itself never sees)
        accepted = not viability_error
        if accepted and cfg.goal.selection == "accept_reject" and results:
            score = sum(r.score for r in results) / len(results)
            if best_score is not None and score < best_score:
                accepted = False
            else:
                best_score = score

        candidate_commit = ""
        try:
            if accepted:
                candidate_commit = snapshot.commit(harness, f"episode {ep}: {cfg.name}")
                last_accepted = candidate_commit
            else:
                # non-destructive rejection: the rejected candidate tree goes into history
                # first (as "candidate N:", outside the episode->commit mapping), then the
                # restore is committed as episode N so the mapping stays gapless
                reason = "viability failed" if viability_error else "rejected"
                candidate_commit = snapshot.commit(
                    harness, f"candidate {ep}: {cfg.name} [{reason}]"
                )
                snapshot.restore(harness, last_accepted)
                snapshot.commit(harness, f"episode {ep}: {cfg.name} [{reason}; rolled back]")
        except Exception as exc:  # noqa: BLE001 - one bad subject must not abort a sweep
            error = f"snapshot failed after episode {ep}: {type(exc).__name__}: {exc}"
            try:
                snapshot.reset_to_checkpoint(harness, last_checkpoint)
            except Exception as restore_exc:  # noqa: BLE001
                error += f"; automatic restore failed: {restore_exc}"
            break
        last_checkpoint = snapshot.head(harness)
        if viability_error and staged_activation:
            pending = _write_pending_candidate(
                cfg.root, commit=candidate_commit, resume_episode=ep + 1,
                reason="viability", error=viability_error,
            )
        else:
            # Accepted evolution advances the active line. A valid but lower-scoring
            # selection rejection deliberately starts over from last-valid rather than
            # inheriting a candidate the configured outer loop rejected.
            pending = None
            _clear_pending_candidate(cfg.root)
        checkpoint_fingerprint = cfg.adapter.disposition_fingerprint(harness)
        done = ep
        for key, value in (res.counters or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value

        eval_history.append({"episode": ep, "accepted": accepted,
                             "results": [r.__dict__ for r in results],
                             "counters": dict(res.counters or {}),
                             "candidate_commit": candidate_commit,
                             "candidate_fingerprint": candidate_fingerprint,
                             "disposition_fingerprint": checkpoint_fingerprint,
                             "disposition_drift": candidate_fingerprint != fingerprint,
                             "failure_kind": "viability" if viability_error else "",
                             "error": viability_error})
        # The snapshot and experiment state are two halves of one durable checkpoint.
        # Persist after every episode, atomically. A crash in the tiny interval after the
        # git commit but before this replace is detected by the strict resume guard above
        # instead of silently resetting selection history.
        _write_json_atomic(history_path, eval_history)
        prior_feedback = cfg.goal.observe_feedback(by_name)  # OBSERVE-visible only
        if viability_error:
            if pending:
                prior_feedback = _repair_feedback(pending, prior_feedback)
            else:
                recovery = (
                    "Your last candidate failed the episode-boundary viability gate and "
                    "was rolled back. Fix the underlying issue in a new candidate: "
                    f"{viability_error[:600]}"
                )
                prior_feedback = (
                    f"{recovery}\n\n{prior_feedback}" if prior_feedback else recovery
                )
        elif prior_feedback and not accepted:
            prior_feedback += "\n(Your last episode's changes were not kept.)"

        if cfg.progress_path is not None:
            if viability_error:
                from proteus.core.adapter import EpisodeResult
                progress_res = EpisodeResult(
                    episode=ep, ok=False, turns=res.turns, error=viability_error,
                    counters=res.counters)
            else:
                progress_res = res
            _append_progress(cfg, ep, progress_res, trace, accepted, results)

    if not history_path.exists():
        _write_json_atomic(history_path, eval_history)
    return RunResult(name=cfg.name, episodes_complete=done, root=str(cfg.root),
                     error=error, eval_history=eval_history, counters=totals)
