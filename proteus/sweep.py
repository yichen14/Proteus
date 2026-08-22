"""Run a grid of self-evolution trajectories — the paper's Step-2 experiment.

A sweep is {arm (disposition)} × {seed}, each a full N-episode trajectory under one
`GoalConfig`. Every run root is kept (the harness is the dependent variable). This is the
harness-agnostic analogue of the research grid: the same sweep runs the minimal harness,
Aki, or any plugged-in adapter, under no-goal or goal.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from proteus.core.adapter import HarnessAdapter
from proteus.core.budget import BUDGET_PROTOCOL_VERSION, PHASES, make_budget_plan
from proteus.core.continuity import PROTOCOL_VERSION
from proteus.core.disposition import Disposition
from proteus.core.episode import RunConfig, completed_episodes, run
from proteus.core.episode_protocol import DEFAULT_EPISODE_PROTOCOL_VERSION
from proteus.core.goal import GoalConfig


MANIFEST_FORMAT_VERSION = 2


class SweepStateError(ValueError):
    """Existing sweep state cannot be safely reused."""


def _json_value(value: Any) -> Any:
    """Return a stable, JSON-safe representation for condition metadata."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_json_value(v) for v in value]
        return sorted(values, key=lambda v: json.dumps(v, sort_keys=True)) \
            if isinstance(value, (set, frozenset)) else values
    raise TypeError(
        f"condition metadata must contain only JSON values and paths, got {type(value).__name__}"
    )


def _sha256_json(value: Any) -> str:
    """Fingerprint a condition value without publishing its literal representation."""
    import hashlib

    encoded = json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _disposition_condition(disposition: Disposition) -> dict:
    """Identify a disposition without publishing its potentially sensitive prompt text."""
    content = {
        "prompt_suffix": disposition.prompt_suffix,
        "per_phase": dict(disposition.per_phase),
        "config": dict(disposition.config),
        "patch": disposition.patch,
    }
    return {"label": disposition.label, "sha256": _sha256_json(content)}


def _sandbox_condition(sandbox: object | None) -> dict | None:
    """Describe an execution boundary without publishing private config values."""
    if sandbox is None:
        return None
    out: dict[str, Any] = {
        "type": f"{type(sandbox).__module__}:{type(sandbox).__qualname__}",
    }
    config = getattr(sandbox, "config", None)
    if config is None:
        return out
    out["config"] = {
        attr: _json_value(getattr(config, attr, None))
        for attr in ("network", "image", "mem_limit", "cpus", "env_passthrough",
                     "entrypoint", "workdir", "user")
    }
    # Literal env values, host mount paths, and arbitrary runner arguments may contain
    # credentials or private paths. Their digests still make resume sensitive to a
    # change without copying those values into the public manifest.
    for attr in ("env", "extra_mounts", "extra_args"):
        value = getattr(config, attr, {})
        out["config"][attr] = {
            "count": len(value),
            "sha256": _sha256_json(value),
        }
    if isinstance(getattr(config, "env", None), Mapping):
        out["config"]["env"]["keys"] = sorted(str(k) for k in config.env)
    return out


def _adapter_condition(adapter: HarnessAdapter) -> dict:
    """Record public runtime knobs while deliberately excluding credentials."""
    surfaces = [{
        "name": s.name,
        "subdir": s.subdir,
        "unit": s.unit,
        "write_tools": sorted(s.write_tools),
        "is_code": s.is_code,
        "free_named": s.free_named,
    } for s in adapter.surfaces()]
    out = {
        "name": adapter.name,
        "type": f"{type(adapter).__module__}:{type(adapter).__qualname__}",
        "continuity_mode": getattr(adapter, "continuity_mode", "native"),
        "staged_activation": bool(getattr(adapter, "staged_activation", False)),
        "surfaces": surfaces,
    }
    runtime = {}
    for attr in ("image", "network", "provider", "model", "base_url",
                 "phase_timeout_s", "permission_mode"):
        value = getattr(adapter, attr, None)
        if value is not None:
            runtime[attr] = _json_value(value)
    sandbox = _sandbox_condition(getattr(adapter, "sandbox", None))
    if sandbox is not None:
        runtime["sandbox"] = sandbox
    if runtime:
        out["runtime"] = runtime
    return out


def _condition(cfg: "SweepConfig", adapter: HarnessAdapter) -> dict:
    from proteus import __version__

    task = None
    if cfg.task is not None:
        task = {
            "id": str(getattr(cfg.task, "id", type(cfg.task).__name__)),
            "base_commit": str(getattr(cfg.task, "base_commit", "")),
        }
    condition = {
        "proteus_version": __version__,
        "default_episode_protocol_version": DEFAULT_EPISODE_PROTOCOL_VERSION,
        "continuity_protocol_version": (
            PROTOCOL_VERSION
            if getattr(adapter, "continuity_mode", "native") == "framework" else None
        ),
        "adapter": _adapter_condition(adapter),
        "arms": [_disposition_condition(arm) for arm in cfg.arms],
        "seeds": cfg.seeds,
        "episodes": cfg.episodes,
        "model": cfg.model,
        "max_turns": cfg.max_turns,
        "min_turns_per_phase": cfg.min_turns_per_phase,
        "announce_budget": cfg.announce_budget,
        "goal": {
            "text": cfg.goal.goal_text(),
            "selection": cfg.goal.selection,
            "evaluators": cfg.goal.describe(),
        },
        "task": task,
        "grader_sandbox": _sandbox_condition(cfg.grader_sandbox),
        "metadata": _json_value(cfg.condition_metadata),
    }
    if cfg.phase_turns or cfg.hard_max_turns or cfg.checkpoint_turns:
        plan = make_budget_plan(
            max_turns=cfg.max_turns,
            min_turns_per_phase=cfg.min_turns_per_phase,
            phase_turns=cfg.phase_turns,
            hard_max_turns=cfg.hard_max_turns,
            checkpoint_turns=cfg.checkpoint_turns,
        )
        condition["budget_protocol"] = {
            "version": BUDGET_PROTOCOL_VERSION,
            "normal_limit": plan.normal_limit,
            "hard_limit": plan.hard_limit,
            "phase_turns": {phase: plan.phase_allowances[phase] for phase in PHASES},
            "checkpoint_turns": plan.checkpoint_turns,
            "unused_priority": "act",
        }
    return condition


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1), encoding="utf-8")
    temporary.replace(path)


def _load_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepStateError(f"cannot resume: manifest {path} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise SweepStateError(f"cannot resume: manifest {path} is not a JSON object")
    return value


@dataclass
class SweepConfig:
    name: str
    adapter_factory: Callable[[], HarnessAdapter]
    arms: Sequence[Disposition]
    seeds: int
    goal: GoalConfig
    root: Path
    model: str = "mock"
    episodes: int = 30
    max_turns: int = 100
    task: object | None = None
    """A `BenchTask` to seed into every run (set automatically when a benchmark
    evaluator is attached)."""
    grader_sandbox: object | None = None
    """Optional isolated grader runner propagated to every benchmark evaluator."""
    min_turns_per_phase: int = 0
    phase_turns: Mapping[str, int] = field(default_factory=dict)
    hard_max_turns: int = 0
    checkpoint_turns: int = 0
    announce_budget: bool = False
    on_existing: str = "refuse"
    """What to do when a run root is already there: "refuse" (default), "resume", or
    "overwrite".

    Run ids are deterministic in (arm, seed), so a second sweep into the same root lands
    on the same directories. Left alone, that silently continues each seed from the
    previous sweep's *evolved* harness rather than from a clean seed, appends a second
    record per seed, and writes a second "episode 1" into the same snapshot history —
    contamination that is invisible in the output. "resume" skips seeds already recorded
    complete and picks a partial one up at the episode after its last snapshot; "overwrite"
    discards the old run roots."""
    condition_metadata: Mapping[str, Any] = field(default_factory=dict)
    """Additional non-secret inputs that define the experiment but cannot be inferred
    from the adapter/evaluator contracts. The CLI records its raw evaluator specs here;
    API callers with configurable custom evaluators should do the same. Resume compares
    this metadata byte-for-byte after canonical JSON normalization."""


def opaque_id(arm: str, seed: int) -> str:
    """Opaque run-dir name: the subject reads its own path, so it must not spell the arm."""
    import hashlib
    return "run-" + hashlib.sha1(f"{arm}:{seed}".encode()).hexdigest()[:12]


def completed_seeds(root: Path, episodes: int) -> set[tuple[str, int]]:
    """(arm, seed) pairs recorded as having finished all `episodes`, from seeds.jsonl."""
    return {(row["arm"], row["seed"]) for row in read_seed_records(root)
            if row.get("episodes_complete", 0) >= episodes and not row.get("error")}


def read_seed_records(root: Path) -> list[dict]:
    """Read the durable last record for each ``(arm, seed)`` in stable order."""
    path = Path(root) / "seeds.jsonl"
    if not path.exists():
        return []
    records: dict[tuple[str, int], dict] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (row["arm"], int(row["seed"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid seed record at {path}:{lineno}: {exc}") from exc
        # assignment preserves first insertion order while replacing the value
        records[key] = row
    return list(records.values())


def _write_seed_record(path: Path, record: dict) -> None:
    records = {(row["arm"], int(row["seed"])): row
               for row in read_seed_records(path.parent)}
    records[(record["arm"], int(record["seed"]))] = record
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row) + "\n" for row in records.values()), encoding="utf-8"
    )
    temporary.replace(path)


def completed_episodes_in(run_root: Path) -> int:
    """Episodes already snapshotted in a run root, without needing its RunConfig."""
    from types import SimpleNamespace
    return completed_episodes(SimpleNamespace(root=run_root))


def run_sweep(cfg: SweepConfig) -> list[dict]:
    if cfg.on_existing not in ("refuse", "resume", "overwrite"):
        raise ValueError(f"on_existing must be refuse/resume/overwrite, got {cfg.on_existing!r}")
    plan = make_budget_plan(
        max_turns=cfg.max_turns,
        min_turns_per_phase=cfg.min_turns_per_phase,
        phase_turns=cfg.phase_turns,
        hard_max_turns=cfg.hard_max_turns,
        checkpoint_turns=cfg.checkpoint_turns,
    )
    if cfg.checkpoint_turns and not cfg.announce_budget:
        raise ValueError("checkpoint_turns requires announce_budget")
    manifest_adapter = cfg.adapter_factory()
    continuity_mode = getattr(manifest_adapter, "continuity_mode", "native")
    if cfg.checkpoint_turns and continuity_mode == "none":
        raise ValueError(
            "checkpoint_turns requires a harness with native or framework continuity"
        )
    cfg.root.mkdir(parents=True, exist_ok=True)
    runs = [{"id": opaque_id(arm.label, s), "arm": arm.label, "seed": s}
            for arm in cfg.arms for s in range(cfg.seeds)]
    manifest_path = cfg.root / "manifest.json"
    records_path = cfg.root / "seeds.jsonl"
    progress_path = cfg.root / "progress"
    runs_path = cfg.root / "runs"
    has_state = (manifest_path.exists() or records_path.exists() or progress_path.exists()
                 or runs_path.exists())

    # Constructing the declaration adapter is side-effect free. Its public runtime knobs
    # form part of the condition; credentials are deliberately never inspected.
    condition = _condition(cfg, manifest_adapter)

    # Validate before mutating anything. Previously even a refused second invocation
    # replaced manifest.json, and resume could silently join episodes run under different
    # goals/models into one trajectory.
    existing_manifest = None
    if cfg.on_existing == "refuse" and has_state:
        raise FileExistsError(
            f"{cfg.root} already holds sweep state. Refusing before changing its manifest; "
            "use a fresh --out, or on_existing='resume' / 'overwrite'."
        )
    if cfg.on_existing == "resume" and has_state:
        if not manifest_path.exists():
            raise SweepStateError(
                f"cannot resume {cfg.root}: existing run state has no manifest.json; "
                "use overwrite or a fresh output directory"
            )
        existing_manifest = _load_manifest(manifest_path)
        previous = existing_manifest.get("condition")
        if existing_manifest.get("format_version") != MANIFEST_FORMAT_VERSION or not isinstance(
            previous, dict
        ):
            raise SweepStateError(
                f"cannot resume {cfg.root}: its manifest predates the v0.2 condition lock. "
                "Finish it with the Proteus version that created it, or use overwrite/new --out."
            )
        if previous != condition:
            changed = sorted(set(previous) | set(condition))
            changed = [key for key in changed if previous.get(key) != condition.get(key)]
            raise SweepStateError(
                f"cannot resume {cfg.root}: experimental condition differs in "
                f"{', '.join(changed)}; use the original configuration or a new --out"
            )

    if cfg.on_existing == "overwrite":
        # Overwrite means the whole sweep, including private records nested below runs/.
        # Clear it before publishing the new manifest so a crash cannot make old runs look
        # as though they belong to the new condition.
        records_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        shutil.rmtree(progress_path, ignore_errors=True)
        shutil.rmtree(runs_path, ignore_errors=True)
        has_state = False

    manifest = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "name": cfg.name, "episodes": cfg.episodes,
        "arms": [a.label for a in cfg.arms], "seeds": cfg.seeds, "runs": runs,
        "model": cfg.model,
        "goal": cfg.goal.goal_text(),
        "evaluators": cfg.goal.describe(),
        "announce_budget": cfg.announce_budget,
        "continuity": {
            "mode": continuity_mode,
            "protocol_version": PROTOCOL_VERSION if continuity_mode == "framework" else None,
        },
        "condition": condition,
    }
    if plan.explicit:
        manifest["budget"] = {
            "protocol_version": BUDGET_PROTOCOL_VERSION,
            "normal_limit": plan.normal_limit,
            "hard_limit": plan.hard_limit,
            "phase_turns": {phase: plan.phase_allowances[phase] for phase in PHASES},
            "checkpoint_turns": plan.checkpoint_turns,
            "unused_priority": "act",
        }
    # Preserve an already-validated resume manifest exactly. A refused or failed resume
    # must be observationally read-only; a new/overwrite sweep publishes atomically.
    if existing_manifest is None:
        _write_json_atomic(manifest_path, manifest)

    done = completed_seeds(cfg.root, cfg.episodes) if cfg.on_existing == "resume" else set()

    records: list[dict] = []
    for arm in cfg.arms:
        for s in range(cfg.seeds):
            rid = opaque_id(arm.label, s)
            run_root = cfg.root / "runs" / rid
            start = 0
            run_root_existed = run_root.exists()
            if run_root_existed:
                if (arm.label, s) in done:
                    continue
                # The refuse and overwrite cases were resolved before publishing the
                # manifest, so an existing target here is a validated resume.
                start = completed_episodes_in(run_root)
                print(f"resuming {arm.label} seed {s} at episode {start + 1}", flush=True)
            rc = RunConfig(
                name=arm.label, adapter=cfg.adapter_factory(), disposition=arm,
                goal=cfg.goal, root=run_root, model=cfg.model,
                episodes=cfg.episodes, max_turns=cfg.max_turns, seed=s,
                min_turns_per_phase=cfg.min_turns_per_phase,
                phase_turns=dict(cfg.phase_turns), hard_max_turns=cfg.hard_max_turns,
                checkpoint_turns=cfg.checkpoint_turns,
                announce_budget=cfg.announce_budget, task=cfg.task,
                grader_sandbox=cfg.grader_sandbox,
                progress_path=cfg.root / "progress" / f"{rid}.jsonl",
            )
            res = run(rc, start=start, resume=run_root_existed)
            rec = {"arm": arm.label, "seed": s, "root": str(run_root),
                   "episodes_complete": res.episodes_complete, "error": res.error,
                   "counters": res.counters}
            records.append(rec)
            _write_seed_record(records_path, rec)
    return records
