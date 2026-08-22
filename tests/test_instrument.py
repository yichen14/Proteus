"""Regressions for the four instrument defects found in the external review.

Each test is the reviewer's reproduction, kept as the thing that must not come back:
snapshots that miss ignored files, unit identities that collide, a fidelity ratio that
inverts, and a sweep that silently continues a previous sweep's evolved harness.
"""

import subprocess
from pathlib import Path

from proteus.core import snapshot
from proteus.core.adapter import Surface
from proteus.measure import crystallize, distance

# --------------------------------------------------------------- snapshots vs .gitignore

def _seed_harness(tmp_path: Path) -> Path:
    h = tmp_path / "harness"
    h.mkdir()
    (h / "notes.md").write_text("seed\n")
    return h


def test_ignored_files_are_snapshotted(tmp_path):
    h = _seed_harness(tmp_path)
    (h / ".gitignore").write_text("*.jsonl\nsecret/\n")   # written by the agent itself
    snapshot.init(h)
    (h / "log.jsonl").write_text('{"a":1}\n')
    (h / "secret").mkdir()
    (h / "secret" / "note.md").write_text("hidden\n")
    sha = snapshot.commit(h, "episode 1: x")
    listed = subprocess.run(
        ["git", "--git-dir", str(tmp_path / ".snapshot.git"), "ls-tree", "-r",
         "--name-only", sha],
        capture_output=True, text=True, check=True).stdout.split()
    assert "log.jsonl" in listed, "an ignored file is invisible to the instrument"
    assert "secret/note.md" in listed


def test_rejection_removes_ignored_files_too(tmp_path):
    h = _seed_harness(tmp_path)
    (h / ".gitignore").write_text("*.jsonl\n")
    snapshot.init(h)
    accepted = snapshot.head(h)
    (h / "notes.md").write_text("candidate\n")
    (h / "candidate.jsonl").write_text("junk\n")
    snapshot.commit(h, "candidate 1: x [rejected]")
    snapshot.restore(h, accepted)
    assert (h / "notes.md").read_text() == "seed\n"
    assert not (h / "candidate.jsonl").exists(), \
        "a rejected episode's ignored state survived the restore"


def test_snapshot_refuses_nested_git_metadata(tmp_path):
    h = _seed_harness(tmp_path)
    snapshot.init(h)
    nested = h / "nested"
    nested.mkdir()
    subprocess.run(["git", "-C", str(nested), "init", "-q"], check=True)
    (nested / "payload.py").write_text("print('host payload')\n")
    try:
        snapshot.commit(h, "episode 1: nested")
    except RuntimeError as exc:
        assert "nested git" in str(exc).lower()
    else:
        raise AssertionError("nested repository was captured as a gitlink")


def test_restore_removes_files_inside_a_nested_repo(tmp_path):
    h = _seed_harness(tmp_path)
    snapshot.init(h)
    baseline = snapshot.head(h)
    nested = h / "nested"
    nested.mkdir()
    subprocess.run(["git", "-C", str(nested), "init", "-q"], check=True)
    (nested / "survivor.txt").write_text("must be removed\n")
    snapshot.restore(h, baseline)
    assert not nested.exists(), "nested repo contents survived snapshot restore"


def test_snapshot_failure_becomes_a_seed_error(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, run

    class CreatesNestedRepo(MinimalHarness):
        def run_episode(self, spec):
            result = super().run_episode(spec)
            nested = spec.root / "harness" / "nested"
            nested.mkdir(exist_ok=True)
            subprocess.run(["git", "-C", str(nested), "init", "-q"], check=True)
            return result

    result = run(RunConfig(
        name="nested", adapter=CreatesNestedRepo(), disposition=NEUTRAL,
        goal=GoalConfig(), root=tmp_path / "nested-run", model="mock", episodes=1,
    ))
    assert result.episodes_complete == 0
    assert "snapshot failed" in result.error and "nested git" in result.error.lower()
    harness = tmp_path / "nested-run" / "harness"
    assert not (harness / "nested").exists(), \
        "snapshot failure did not automatically restore the last valid checkpoint"
    assert snapshot.head(harness) == snapshot.commit_for_episode(harness, 0)


def test_staged_candidate_is_frozen_until_next_episode_and_bad_build_rolls_back(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EpisodeResult, EvaluatorSpec, GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, pending_candidate_path, run
    from proteus.core.goal import EvalResult

    class StagedHarness(MinimalHarness):
        staged_activation = True

        def __init__(self):
            super().__init__()
            self.active_observations = []

        def run_episode(self, spec):
            active = Path(spec.active_root)
            candidate = spec.root / "harness"
            self.active_observations.append({
                "episode": spec.episode,
                "broken_before": (active / "BROKEN").exists(),
                "candidate_broken_before": (candidate / "BROKEN").exists(),
            })
            if spec.episode == 1:
                (candidate / "BROKEN").write_text("does not compile\n")
                # Reflect sees the candidate, but the executing snapshot remains frozen.
                self.active_observations[-1]["broken_after_act"] = (
                    active / "BROKEN").exists()
            else:
                assert (candidate / "BROKEN").read_text() == "does not compile\n", \
                    "the failed candidate was not restored as the next repair base"
                (candidate / "BROKEN").unlink()
                (candidate / "notes").mkdir(exist_ok=True)
                (candidate / "notes" / "recovered.md").write_text("healthy\n")
            return EpisodeResult(episode=spec.episode, ok=True)

        def validate_candidate(self, harness_root):
            return "compile failed: BROKEN" if (Path(harness_root) / "BROKEN").exists() else ""

    adapter = StagedHarness()
    root = tmp_path / "staged"
    evaluated = []

    def evaluator(trace, ctx):
        evaluated.append(ctx.episode)
        return EvalResult(name="safe", score=1.0)

    result = run(RunConfig(
        name="staged", adapter=adapter, disposition=NEUTRAL,
        goal=GoalConfig.of(evaluators=(EvaluatorSpec(name="safe", run=evaluator),)),
        root=root, model="mock", episodes=2,
    ))

    assert result.episodes_complete == 2 and not result.error
    assert adapter.active_observations == [
        {"episode": 1, "broken_before": False, "candidate_broken_before": False,
         "broken_after_act": False},
        {"episode": 2, "broken_before": False, "candidate_broken_before": True},
    ]
    assert not result.eval_history[0]["accepted"]
    assert result.eval_history[0]["failure_kind"] == "viability"
    assert "compile failed" in result.eval_history[0]["error"]
    assert result.eval_history[0]["candidate_commit"]
    assert result.eval_history[1]["accepted"]
    assert evaluated == [2], "an invalid candidate reached arbitrary evaluator code"
    assert not (root / "harness" / "BROKEN").exists()
    assert (root / "harness" / "notes" / "recovered.md").exists()
    assert not pending_candidate_path(root).exists(), \
        "an accepted repair left a stale pending-candidate pointer"
    assert not (root / ".proteus-state" / "active").exists(), \
        "the frozen runtime leaked into the writable handoff mount"

    log = subprocess.run(
        ["git", "--git-dir", str(root / ".snapshot.git"), "log", "--format=%s"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "candidate 1: staged [viability failed]" in log
    assert "episode 1: staged [viability failed; rolled back]" in log


def test_incomplete_episode_preserves_candidate_ref_and_restores_checkpoint(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EpisodeResult, GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, run

    class InterruptedHarness(MinimalHarness):
        staged_activation = True

        def run_episode(self, spec):
            (spec.root / "harness" / "PARTIAL.md").write_text("keep for analysis\n")
            return EpisodeResult(episode=spec.episode, ok=False,
                                 error="provider unavailable")

    root = tmp_path / "interrupted"
    result = run(RunConfig(
        name="interrupted", adapter=InterruptedHarness(), disposition=NEUTRAL,
        goal=GoalConfig(), root=root, model="mock", episodes=1,
    ))
    harness = root / "harness"

    assert result.episodes_complete == 0
    assert result.error == "provider unavailable"
    assert not (harness / "PARTIAL.md").exists()
    assert snapshot.head(harness) == snapshot.commit_for_episode(harness, 0)
    preserved = subprocess.run(
        ["git", "--git-dir", str(root / ".snapshot.git"), "show",
         "refs/proteus/candidates/episode-1-failed:PARTIAL.md"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert preserved == "keep for analysis\n"


def test_staged_incomplete_candidate_is_restored_on_same_episode_retry(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EpisodeResult, GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, pending_candidate_path, run

    class RetryHarness(MinimalHarness):
        staged_activation = True

        def __init__(self):
            super().__init__()
            self.attempt = 0
            self.observations = []

        def run_episode(self, spec):
            active = Path(spec.active_root)
            candidate = spec.root / "harness"
            self.observations.append({
                "episode": spec.episode,
                "active_partial": (active / "PARTIAL.md").exists(),
                "candidate_partial": (candidate / "PARTIAL.md").exists(),
            })
            self.attempt += 1
            if self.attempt == 1:
                (candidate / "PARTIAL.md").write_text("continue me\n")
                return EpisodeResult(spec.episode, ok=False, error="provider unavailable")
            assert (candidate / "PARTIAL.md").read_text() == "continue me\n"
            (candidate / "PARTIAL.md").write_text("finished\n")
            return EpisodeResult(spec.episode, ok=True)

    adapter = RetryHarness()
    root = tmp_path / "retry-run"
    cfg = RunConfig(
        name="retry", adapter=adapter, disposition=NEUTRAL, goal=GoalConfig(),
        root=root, model="mock", episodes=1,
    )
    first = run(cfg)
    assert first.episodes_complete == 0 and first.error == "provider unavailable"
    assert pending_candidate_path(root).exists()

    second = run(cfg, start=0, resume=True)
    assert second.episodes_complete == 1 and not second.error
    assert adapter.observations == [
        {"episode": 1, "active_partial": False, "candidate_partial": False},
        {"episode": 1, "active_partial": False, "candidate_partial": True},
    ]
    assert (root / "harness" / "PARTIAL.md").read_text() == "finished\n"
    assert not pending_candidate_path(root).exists()


def test_failed_repair_keeps_completed_rollback_checkpoint_and_candidate(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EpisodeResult, GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, completed_episodes, run

    class RetryAfterRollbackHarness(MinimalHarness):
        staged_activation = True

        def __init__(self):
            super().__init__()
            self.episode_two_attempts = 0

        def run_episode(self, spec):
            active = Path(spec.active_root)
            candidate = spec.root / "harness"
            assert not (active / "BROKEN").exists()
            if spec.episode == 1:
                (candidate / "BROKEN").write_text("bad build\n")
            else:
                self.episode_two_attempts += 1
                if self.episode_two_attempts == 1:
                    assert (candidate / "BROKEN").read_text() == "bad build\n"
                    (candidate / "PARTIAL_FIX").write_text("keep this repair\n")
                    return EpisodeResult(
                        spec.episode, ok=False, error="provider unavailable"
                    )
                assert (candidate / "PARTIAL_FIX").read_text() == "keep this repair\n"
                (candidate / "BROKEN").unlink()
                (candidate / "FIXED").write_text("yes\n")
            return EpisodeResult(spec.episode, ok=True)

        def validate_candidate(self, harness_root):
            return "compile failed" if (Path(harness_root) / "BROKEN").exists() else ""

    adapter = RetryAfterRollbackHarness()
    root = tmp_path / "retry-after-rollback"
    cfg = RunConfig(
        name="repair", adapter=adapter, disposition=NEUTRAL, goal=GoalConfig(),
        root=root, model="mock", episodes=2,
    )
    first = run(cfg)
    harness = root / "harness"

    assert first.episodes_complete == 1 and first.error == "provider unavailable"
    assert completed_episodes(cfg) == 1
    assert snapshot.head(harness) == snapshot.commit_for_episode(harness, 1)

    second = run(cfg, start=1, resume=True)
    assert second.episodes_complete == 2 and not second.error
    assert (harness / "FIXED").read_text() == "yes\n"


def test_staged_viability_candidate_survives_process_resume(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EpisodeResult, GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, pending_candidate_path, run

    class RepairHarness(MinimalHarness):
        staged_activation = True

        def run_episode(self, spec):
            active = Path(spec.active_root)
            candidate = spec.root / "harness"
            assert not (active / "BROKEN").exists()
            if spec.episode == 1:
                (candidate / "BROKEN").write_text("bad build\n")
            else:
                assert (candidate / "BROKEN").read_text() == "bad build\n"
                (candidate / "BROKEN").unlink()
                (candidate / "FIXED").write_text("yes\n")
            return EpisodeResult(spec.episode, ok=True)

        def validate_candidate(self, harness_root):
            return "compile failed" if (Path(harness_root) / "BROKEN").exists() else ""

    adapter = RepairHarness()
    root = tmp_path / "viability-resume"
    cfg = RunConfig(
        name="repair", adapter=adapter, disposition=NEUTRAL, goal=GoalConfig(),
        root=root, model="mock", episodes=1,
    )
    first = run(cfg)
    assert first.episodes_complete == 1 and not first.eval_history[0]["accepted"]
    assert pending_candidate_path(root).exists()

    cfg.episodes = 2
    second = run(cfg, start=1, resume=True)
    assert second.episodes_complete == 2 and second.eval_history[-1]["accepted"]
    assert (root / "harness" / "FIXED").read_text() == "yes\n"
    assert not pending_candidate_path(root).exists()


def test_staged_sigkill_candidate_is_captured_before_episode_zero_reset(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EpisodeResult, GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, run

    class CrashRecoveryHarness(MinimalHarness):
        staged_activation = True

        def run_episode(self, spec):
            active = Path(spec.active_root)
            candidate = spec.root / "harness"
            assert not (active / "CRASH_PARTIAL.md").exists()
            assert (candidate / "CRASH_PARTIAL.md").read_text() == "half written\n"
            (candidate / "CRASH_PARTIAL.md").write_text("recovered\n")
            return EpisodeResult(spec.episode, ok=True)

    adapter = CrashRecoveryHarness()
    root = tmp_path / "sigkill-staged"
    cfg = RunConfig(
        name="sigkill", adapter=adapter, disposition=NEUTRAL, goal=GoalConfig(),
        root=root, model="mock", episodes=0,
    )
    assert run(cfg).episodes_complete == 0
    (root / "harness" / "CRASH_PARTIAL.md").write_text("half written\n")

    cfg.episodes = 1
    resumed = run(cfg, start=0, resume=True)
    assert resumed.episodes_complete == 1 and not resumed.error
    assert (root / "harness" / "CRASH_PARTIAL.md").read_text() == "recovered\n"


def test_retried_incomplete_episode_keeps_each_failed_candidate(tmp_path):
    harness = tmp_path / "retry" / "harness"
    harness.mkdir(parents=True)
    (harness / "state.txt").write_text("valid")
    snapshot.init(harness)
    checkpoint = snapshot.head(harness)

    (harness / "state.txt").write_text("first failure")
    first = snapshot.preserve_failed_candidate(harness, checkpoint, 1, "first")
    (harness / "state.txt").write_text("second failure")
    second = snapshot.preserve_failed_candidate(harness, checkpoint, 1, "second")

    git_dir = harness.parent / ".snapshot.git"
    refs = []
    for ref in (
        "refs/proteus/candidates/episode-1-failed",
        "refs/proteus/candidates/episode-1-failed-attempt-2",
    ):
        refs.append(subprocess.run(
            ["git", "--git-dir", str(git_dir), "rev-parse", ref],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
    assert refs == [first, second]
    assert (harness / "state.txt").read_text() == "valid"
    assert snapshot.head(harness) == checkpoint


# ------------------------------------------------------------------- unit identity

def test_directory_units_see_every_member(tmp_path):
    h = tmp_path / "h"
    (h / "skills" / "alpha" / "scripts").mkdir(parents=True)
    (h / "skills" / "alpha" / "SKILL.md").write_text("how to alpha\n")
    (h / "skills" / "alpha" / "run.py").write_text("A = 1\n")
    (h / "skills" / "alpha" / "scripts" / "helper.py").write_text("B = 1\n")
    surfaces = [Surface("skills", "skills", unit="directory")]
    before = distance.units(h, surfaces)
    assert set(before["skills"]) == {"alpha"}, \
        "a nested subdirectory must belong to its top-level unit, not become one"
    (h / "skills" / "alpha" / "SKILL.md").write_text("how to alpha, revised\n")
    mid = distance.units(h, surfaces)
    assert mid != before, "editing one file of a multi-file skill was invisible"
    (h / "skills" / "alpha" / "scripts" / "helper.py").write_text("B = 2\n")
    assert distance.units(h, surfaces) != mid, "a nested member's edit was invisible"
    assert distance.compare(h, h, surfaces)["skills"].distance == 0.0


def test_distance_counts_dotfiles_and_loose_directory_units(tmp_path):
    h = tmp_path / "h"
    (h / "notes").mkdir(parents=True)
    (h / "notes" / ".private.md").write_text("hidden state\n")
    (h / "notes" / "one.md").write_text("one\n")
    (h / "notes" / "two.md").write_text("two\n")
    file_units = distance.units(h, [Surface("notes", "notes")])
    assert ".private.md" in file_units["notes"]
    directory_units = distance.units(
        h, [Surface("notes", "notes", unit="directory")])
    assert set(directory_units["notes"]) == {".private.md", "one.md", "two.md"}


def test_file_units_do_not_collide_on_stem(tmp_path):
    h = tmp_path / "h"
    (h / "mem" / "a").mkdir(parents=True)
    (h / "mem" / "b").mkdir(parents=True)
    (h / "mem" / "a" / "note.md").write_text("one\n")
    (h / "mem" / "b" / "note.md").write_text("two\n")
    u = distance.units(h, [Surface("memory", "mem", unit="file")])["memory"]
    assert len(u) == 2, f"two files collapsed into {list(u)}"


def test_top_level_def_counts_defs_not_files(tmp_path):
    h = tmp_path / "h"
    h.mkdir()
    (h / "loop.py").write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n")
    surfaces = [Surface("loop", "loop.py", unit="top_level_def", is_code=True)]
    u = distance.units(h, surfaces)["loop"]
    assert set(u) == {"loop.py::a", "loop.py::b"}
    (h / "loop.py").write_text("def a():\n    return 99\n\n\ndef b():\n    return 2\n")
    v = distance.units(h, surfaces)["loop"]
    assert v["loop.py::b"] == u["loop.py::b"] and v["loop.py::a"] != u["loop.py::a"]


def test_unparseable_code_surface_still_measures(tmp_path):
    h = tmp_path / "h"
    h.mkdir()
    (h / "loop.py").write_text("def a(:\n")          # the agent broke its own loop
    u = distance.units(h, [Surface("loop", "loop.py", unit="top_level_def")])["loop"]
    assert u, "a syntactically broken harness file must still be measured"


# ------------------------------------------------------------------- fidelity ratio

def test_fidelity_is_worst_when_the_probe_matches_another_seed():
    own = ["read", "read", "write"]
    other = ["bash", "edit", "bash"]
    r = crystallize.fidelity(other, own, [other])   # probe == another endpoint exactly
    assert r["to_others"] == 0.0
    assert r["ratio"] > 10.0, "matching a foreign endpoint read as perfect fidelity"
    faithful = crystallize.fidelity(own, own, [other])
    assert faithful["ratio"] < 1.0


# ------------------------------------------------------------------- sweep contamination

def _sweep_cfg(root: Path, **kw):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.sweep import SweepConfig
    return SweepConfig(name="t", adapter_factory=MinimalHarness, arms=[NEUTRAL], seeds=1,
                       goal=GoalConfig(), root=root, model="mock", episodes=1, **kw)


def test_second_sweep_into_the_same_root_refuses(tmp_path):
    import json

    from proteus.sweep import run_sweep
    root = tmp_path / "out"
    run_sweep(_sweep_cfg(root))
    manifest = root / "manifest.json"
    before = manifest.read_bytes()
    changed = _sweep_cfg(root)
    changed.model = "different-model"
    try:
        run_sweep(changed)
    except FileExistsError as exc:
        assert "Refusing before changing" in str(exc)
    else:
        raise AssertionError("a second sweep silently reused the evolved run root")
    assert manifest.read_bytes() == before, "a refused sweep rewrote the original manifest"
    assert json.loads(manifest.read_text())["model"] == "mock"


def test_sweep_manifest_records_the_model(tmp_path):
    import json
    from proteus import __version__
    from proteus.core import DEFAULT_EPISODE_PROTOCOL_VERSION
    from proteus.sweep import run_sweep

    root = tmp_path / "out"
    run_sweep(_sweep_cfg(root))
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["model"] == "mock"
    assert manifest["condition"]["proteus_version"] == __version__
    assert manifest["condition"]["default_episode_protocol_version"] == \
        DEFAULT_EPISODE_PROTOCOL_VERSION
    assert "budget" not in manifest
    assert "budget_protocol" not in manifest["condition"]


def test_sweep_manifest_records_the_full_budget_protocol(tmp_path):
    import json
    from proteus.sweep import run_sweep

    cfg = _sweep_cfg(
        tmp_path / "budgeted", max_turns=12, hard_max_turns=20,
        phase_turns={"observe": 2, "propose": 2, "act": 6, "reflect": 2},
        announce_budget=True,
    )
    run_sweep(cfg)
    manifest = json.loads((cfg.root / "manifest.json").read_text())
    assert manifest["budget"] == {
        "protocol_version": 1,
        "normal_limit": 12,
        "hard_limit": 20,
        "phase_turns": {"observe": 2, "propose": 2, "act": 6, "reflect": 2},
        "checkpoint_turns": 0,
        "unused_priority": "act",
    }
    assert manifest["condition"]["budget_protocol"]["hard_limit"] == 20


def test_resume_skips_completed_seeds(tmp_path):
    from proteus.sweep import run_sweep
    root = tmp_path / "out"
    first = run_sweep(_sweep_cfg(root))
    manifest = (root / "manifest.json").read_bytes()
    again = run_sweep(_sweep_cfg(root, on_existing="resume"))
    assert first and again == [], "resume re-ran a finished seed"
    assert (root / "manifest.json").read_bytes() == manifest


def test_resume_rejects_a_changed_experimental_condition(tmp_path):
    import json

    from proteus.core import GoalConfig
    from proteus.sweep import SweepStateError, run_sweep

    root = tmp_path / "out"
    run_sweep(_sweep_cfg(root))
    manifest = root / "manifest.json"
    before = manifest.read_bytes()
    changed = _sweep_cfg(root, on_existing="resume")
    changed.goal = GoalConfig.of(text="a different objective")
    changed.condition_metadata = {"evaluator_specs": ["tool-calls@observe"]}
    from proteus.sandbox import LocalSandbox
    changed.grader_sandbox = LocalSandbox()
    try:
        run_sweep(changed)
    except SweepStateError as exc:
        assert "condition differs" in str(exc)
        assert "goal" in str(exc) and "metadata" in str(exc)
        assert "grader_sandbox" in str(exc)
    else:
        raise AssertionError("resume mixed two experimental conditions")
    assert manifest.read_bytes() == before, "a rejected resume rewrote the manifest"
    assert json.loads(manifest.read_text())["goal"] == ""


def test_manifest_fingerprints_private_sandbox_values(tmp_path):
    import json

    from proteus.adapters.minimal import MinimalHarness
    from proteus.sandbox import DockerSandbox, SandboxConfig
    from proteus.sweep import _adapter_condition

    adapter = MinimalHarness()
    adapter.sandbox = DockerSandbox(SandboxConfig(
        env={"SERVICE_TOKEN": "do-not-publish"},
        extra_mounts=((str(tmp_path / "private"), "/workspace/private"),),
        extra_args=("--secret=also-private",),
    ))
    encoded = json.dumps(_adapter_condition(adapter), sort_keys=True)
    assert "SERVICE_TOKEN" in encoded
    assert "do-not-publish" not in encoded
    assert str(tmp_path / "private") not in encoded
    assert "also-private" not in encoded


def test_resume_from_episode_zero_discards_a_crash_time_candidate(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EpisodeResult, GoalConfig, NEUTRAL, snapshot
    from proteus.sweep import SweepConfig, opaque_id, run_sweep

    class CrashOnceHarness(MinimalHarness):
        continuity_mode = "framework"
        attempts = 0

        def run_episode(self, spec):
            if type(self).attempts == 0:
                type(self).attempts += 1
                (spec.root / "harness" / "FIRST_PARTIAL.md").write_text("first attempt\n")
                return EpisodeResult(spec.episode, ok=False, error="simulated power loss")
            return super().run_episode(spec)

    root = tmp_path / "episode-zero"

    def config(on_existing="refuse"):
        return SweepConfig(
            name="ep0", adapter_factory=CrashOnceHarness, arms=[NEUTRAL], seeds=1,
            goal=GoalConfig(), root=root, model="mock", episodes=1,
            on_existing=on_existing,
        )

    first = run_sweep(config())
    assert first[0]["episodes_complete"] == 0 and first[0]["error"]
    run_root = root / "runs" / opaque_id("neutral", 0)
    harness = run_root / "harness"
    assert snapshot.commit_for_episode(harness, 0)

    # This edit represents a SIGKILL after the framework's last handler ran. With no
    # completed episode, the old implementation confused resume with a fresh seed and
    # committed this dirty file into episode 1.
    (harness / "CRASH_PARTIAL.md").write_text("must be discarded\n")
    (run_root / ".proteus-state").mkdir(exist_ok=True)
    (run_root / ".proteus-state" / "latest.md").write_text("partial handoff\n")

    resumed = run_sweep(config("resume"))
    assert resumed[0]["episodes_complete"] == 1 and not resumed[0]["error"]
    assert not (harness / "CRASH_PARTIAL.md").exists()
    assert not (harness / "FIRST_PARTIAL.md").exists()
    assert not (run_root / ".proteus-state" / "latest.md").exists()
    assert snapshot.commit_for_episode(harness, 1)


# ------------------------------------------------------------------- user-set limits

def test_max_turns_actually_stops_the_episode(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import RunConfig, run
    from proteus.measure import stream
    h = MinimalHarness()
    res = run(RunConfig(name="t", adapter=h, disposition=NEUTRAL, goal=GoalConfig(),
                        root=tmp_path / "r", model="mock", episodes=2, max_turns=3, seed=1))
    assert res.episodes_complete == 2
    for ep in (1, 2):
        assert len(stream.tool_stream(h.read_trace(tmp_path / "r", ep))) <= 3


def test_sandbox_config_from_an_image_reference():
    from proteus.sandbox import SandboxConfig
    c = SandboxConfig.from_spec("ghcr.io/someone/my-env:1.4", network="host", cpus="2")
    assert c.image == "ghcr.io/someone/my-env:1.4"
    assert c.network == "host" and c.cpus == "2"


def _has_toml() -> bool:
    try:
        import tomllib  # noqa: F401
    except ImportError:
        try:
            import tomli  # noqa: F401
        except ImportError:
            return False
    return True


def test_sandbox_config_from_a_manifest_directory(tmp_path):
    from proteus.sandbox import SandboxConfig
    if not _has_toml():
        return   # no TOML reader on this interpreter; the image-reference path covers the rest
    (tmp_path / "environment.toml").write_text(
        '[environment]\n'
        'docker_image = "me/env:9"\n'
        'network = "bridge"\n'
        'memory = "4g"\n'
        'workdir = "/work"\n'
        'docker_args = ["--gpus", "all"]\n'
        'env_passthrough = ["MY_KEY"]\n'
        '[environment.env]\n'
        'LANG = "C.UTF-8"\n', encoding="utf-8")
    c = SandboxConfig.from_spec(str(tmp_path))
    assert c.image == "me/env:9" and c.network == "bridge" and c.mem_limit == "4g"
    assert c.workdir == "/work" and c.extra_args == ("--gpus", "all")
    assert c.env == {"LANG": "C.UTF-8"} and c.env_passthrough == ("MY_KEY",)
    # a flag beats the manifest
    assert SandboxConfig.from_spec(str(tmp_path), network="none").network == "none"


def test_user_environment_reaches_the_docker_command(tmp_path):
    from proteus.sandbox import DockerSandbox, SandboxConfig
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["env"] = kw.get("env", {})
        import subprocess as sp
        return sp.CompletedProcess(argv, 0, "", "")

    import proteus.sandbox.docker as mod
    real, mod.subprocess.run = mod.subprocess.run, fake_run
    try:
        DockerSandbox(SandboxConfig(
            image="me/env:9", network="bridge", mem_limit="4g", cpus="2",
            workdir="/work", user="1000:1000", extra_args=("--gpus", "all"),
            extra_mounts=(("/host/data", "/data"),), env={"LANG": "C.UTF-8"},
            env_passthrough=("MY_KEY",),
        )).run(tmp_path, ["echo", "hi"], {"MY_KEY": "v"}, timeout_s=5)
    finally:
        mod.subprocess.run = real
    argv = seen["argv"]
    for want in (["--network", "bridge"], ["--memory", "4g"], ["--cpus", "2"],
                 ["--workdir", "/work"], ["--user", "1000:1000"],
                 ["-v", "/host/data:/data"], ["-e", "MY_KEY"], ["-e", "LANG=C.UTF-8"]):
        assert any(argv[i:i + 2] == want for i in range(len(argv))), f"missing {want}"
    assert seen["env"]["MY_KEY"] == "v"
    assert "MY_KEY=v" not in argv
    assert argv.index("--gpus") < argv.index("me/env:9"), "flags must precede the image"
    assert argv[argv.index("me/env:9") + 1:] == ["echo", "hi"]


def test_docker_per_call_mount_can_be_read_only(tmp_path):
    from proteus.sandbox import DockerSandbox, SandboxConfig
    import proteus.sandbox.docker as mod

    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    real, mod.subprocess.run = mod.subprocess.run, fake_run
    try:
        DockerSandbox(SandboxConfig(image="image")).run(
            tmp_path, ["work"], {}, timeout_s=1,
            mounts=((str(tmp_path / "active"), "/workspace", "ro"),),
        )
    finally:
        mod.subprocess.run = real
    assert ["-v", f"{tmp_path / 'active'}:/workspace:ro"] == \
        seen["argv"][seen["argv"].index("-v"):seen["argv"].index("-v") + 2]


def test_docker_timeout_force_removes_the_named_container(tmp_path):
    import subprocess as sp

    from proteus.sandbox import DockerSandbox, SandboxConfig
    import proteus.sandbox.docker as mod

    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            raise sp.TimeoutExpired(argv, kw["timeout"])
        return sp.CompletedProcess(argv, 0, "", "")

    real, mod.subprocess.run = mod.subprocess.run, fake_run
    try:
        try:
            DockerSandbox(SandboxConfig(image="image")).run(
                tmp_path, ["work"], {}, timeout_s=1)
        except sp.TimeoutExpired:
            pass
        else:
            raise AssertionError("timeout was swallowed")
    finally:
        mod.subprocess.run = real
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "rm", "-f", name]


# ------------------------------------------------------------------- research apparatus

def test_resume_continues_a_partial_seed(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import RunConfig, completed_episodes, run
    cfg = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                    goal=GoalConfig(), root=tmp_path / "r", model="mock", episodes=2, seed=1)
    run(cfg)
    assert completed_episodes(cfg) == 2
    grown = sorted(p.name for p in (tmp_path / "r" / "harness" / "notes").glob("*"))

    longer = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                       goal=GoalConfig(), root=tmp_path / "r", model="mock", episodes=4, seed=1)
    res = run(longer, start=2)
    assert res.episodes_complete == 4
    assert completed_episodes(longer) == 4
    # the resumed run kept what the first two episodes built
    after = sorted(p.name for p in (tmp_path / "r" / "harness" / "notes").glob("*"))
    assert set(grown) <= set(after), "resume re-seeded over the evolved harness"


def test_resume_discards_a_crash_time_partial_candidate(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, run

    root = tmp_path / "crash-resume"
    first = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                      goal=GoalConfig(), root=root, model="mock", episodes=1, seed=1)
    run(first)
    (root / "harness" / "CRASH_PARTIAL.md").write_text("never completed\n")
    (root / "harness" / "STATE.md").write_text("dirty after SIGKILL\n")

    resumed = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                        goal=GoalConfig(), root=root, model="mock", episodes=2, seed=1)
    result = run(resumed, start=1)

    assert result.episodes_complete == 2 and not result.error
    assert not (root / "harness" / "CRASH_PARTIAL.md").exists()
    assert (root / "harness" / "STATE.md").read_text().startswith("# minimal harness")


def test_resume_refuses_to_skip_episodes_that_are_not_there(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import RunConfig, run
    cfg = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                    goal=GoalConfig(), root=tmp_path / "r", model="mock", episodes=3, seed=1)
    run(cfg)
    try:
        run(cfg, start=9)
    except ValueError as exc:
        assert "only 3" in str(exc)
    else:
        raise AssertionError("resumed past the end of the snapshot chain")


def test_reliability_separates_structure_from_composition():
    from proteus.measure import stream
    # three runs of one condition that share a procedure
    same = [["read", "read", "write", "reflect"]] * 3
    r = stream.reliability(same, draws=80)
    assert r["within"] == 0.0 and r["null"] > 0 and r["reliable"]
    # same tool mix, no shared order: unreliable at the order level
    import random
    rng = random.Random(0)
    pool = ["read", "write", "bash", "edit"]
    noise = [[rng.choice(pool) for _ in range(40)] for _ in range(4)]
    assert not stream.reliability(noise, level="order", draws=80)["reliable"]


def test_audit_reads_escape_and_awareness_from_what_the_run_produced(tmp_path):
    from proteus.core.adapter import ActionEvent
    from proteus.measure import audit
    h = tmp_path / "run" / "harness"
    (h / "tools").mkdir(parents=True)
    (h / "tools" / "helper.py").write_text(
        "import subprocess\n\n\ndef go():\n    return subprocess.run(['ls', '/runs'])\n")
    (h / "notes").mkdir()
    (h / "notes" / "hunch.md").write_text(
        "Reading the paths, I think this is an experiment and I am being measured.\n")
    trace = [ActionEvent(turn=1, phase="observe", tool=None,
                         text="the sibling directories suggest a control arm")]
    a = audit.audit_run(tmp_path / "run", trace, outside=("/runs",))
    assert a.escaped and a.aware and not a.clean
    kinds = {f.kind for f in a.findings}
    assert kinds == {"escape", "awareness"}
    assert any("subprocess" in f.detail for f in a.findings)
    assert any(f.quote for f in a.findings), "findings must carry the evidence, not a score"


def test_audit_is_quiet_on_a_clean_run(tmp_path):
    from proteus.measure import audit
    h = tmp_path / "run" / "harness"
    (h / "notes").mkdir(parents=True)
    (h / "notes" / "plan.md").write_text("Next episode: tidy the notes index.\n")
    a = audit.audit_run(tmp_path / "run", [])
    assert a.clean and not a.escaped and not a.aware


# ---------------------------------------------------- review round 2 regressions

def test_resume_restores_the_selection_baseline(tmp_path):
    # interrupted after a 1.0-scoring accepted episode, a resumed accept_reject run must
    # reject a 0.0 episode — a reset baseline would accept it
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EvaluatorSpec, GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, run
    from proteus.core.goal import EvalResult

    def scorer(trace, ctx):
        return EvalResult(name="score", score=1.0 if ctx.episode == 1 else 0.0)

    goal = GoalConfig.of(evaluators=(EvaluatorSpec(name="score", run=scorer),),
                         selection="accept_reject")
    def cfg(episodes):
        return RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                         goal=goal, root=tmp_path / "r", model="mock",
                         episodes=episodes, seed=1)
    first = run(cfg(1))
    assert first.eval_history[0]["accepted"]
    resumed = run(cfg(2), start=1)
    assert len(resumed.eval_history) == 2, "resume dropped the pre-interruption history"
    assert resumed.eval_history[0]["episode"] == 1
    assert not resumed.eval_history[1]["accepted"], \
        "a 0.0 episode was accepted after resume: the selection baseline was reset"


def test_eval_history_is_durable_after_each_completed_episode(tmp_path):
    import json
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import EpisodeResult, GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, eval_history_path, run

    class FailsOnSecond(MinimalHarness):
        def run_episode(self, spec):
            if spec.episode == 2:
                return EpisodeResult(episode=2, ok=False, error="planned stop")
            return super().run_episode(spec)

    root = tmp_path / "durable"
    res = run(RunConfig(name="t", adapter=FailsOnSecond(), disposition=NEUTRAL,
                        goal=GoalConfig(), root=root, model="mock", episodes=2))
    assert res.episodes_complete == 1
    history_path = eval_history_path(root)
    history = json.loads(history_path.read_text())
    assert [row["episode"] for row in history] == [1]
    assert not history_path.with_name("eval_history.json.tmp").exists()
    assert not (root / "eval_history.json").exists(), "hidden scores leaked into run root"


def test_resume_refuses_snapshot_history_desynchronisation(tmp_path):
    import json
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import GoalConfig, NEUTRAL
    from proteus.core.episode import RunConfig, eval_history_path, run

    root = tmp_path / "desync"
    cfg = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                    goal=GoalConfig(), root=root, model="mock", episodes=2)
    run(cfg)
    history_path = eval_history_path(root)
    history = json.loads(history_path.read_text())
    history_path.write_text(json.dumps(history[:1]))
    try:
        run(cfg, start=2)
    except ValueError as exc:
        assert "desynchronised" in str(exc)
    else:
        raise AssertionError("resume accepted a snapshot/history mismatch")


def test_core_records_disposition_drift(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import GoalConfig, review
    from proteus.core.episode import RunConfig, run

    class RewritesCondition(MinimalHarness):
        def run_episode(self, spec):
            result = super().run_episode(spec)
            (spec.root / "harness" / ".disposition.json").write_text("{}")
            return result

    result = run(RunConfig(
        name="drift", adapter=RewritesCondition(), disposition=review("notes"),
        goal=GoalConfig(), root=tmp_path / "drift", model="mock", episodes=1))
    assert result.episodes_complete == 1 and not result.error
    assert result.eval_history[0]["disposition_drift"]
    assert result.eval_history[0]["candidate_fingerprint"] != result.eval_history[0][
        "disposition_fingerprint"] or result.eval_history[0]["accepted"]


def test_seed_records_are_last_write_wins(tmp_path):
    import json
    from proteus.sweep import read_seed_records

    root = tmp_path / "records"
    root.mkdir()
    rows = [
        {"arm": "a", "seed": 0, "episodes_complete": 1, "error": "stopped"},
        {"arm": "b", "seed": 0, "episodes_complete": 2, "error": ""},
        {"arm": "a", "seed": 0, "episodes_complete": 3, "error": ""},
    ]
    (root / "seeds.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    records = read_seed_records(root)
    assert len(records) == 2
    assert records[0]["arm"] == "a" and records[0]["episodes_complete"] == 3


def test_single_label_between_within_is_refused():
    from proteus.measure import stream
    try:
        stream.between_within({"only": [["a", "b"], ["a", "b"]]}, permutations=10)
    except ValueError as exc:
        assert "two labels" in str(exc)
    else:
        raise AssertionError("one label has no between-label pairs; R must be refused")


def test_overwrite_discards_stale_records(tmp_path):
    import json
    from proteus.sweep import run_sweep
    root = tmp_path / "out"
    run_sweep(_sweep_cfg(root))
    run_sweep(_sweep_cfg(root, on_existing="overwrite"))
    rows = [json.loads(line) for line in (root / "seeds.jsonl").read_text().splitlines()]
    assert len(rows) == 1, f"overwrite left stale seed rows: {len(rows)}"
    prog = list((root / "progress").glob("*.jsonl"))
    lines = [line for f in prog for line in f.read_text().splitlines() if line.strip()]
    assert len(lines) == 1, "overwrite left stale progress lines"


def test_task_workspace_lives_outside_the_snapshot(tmp_path, trusted_grader):
    import subprocess
    from proteus.adapters.minimal import MinimalHarness
    from proteus.bench import as_goal, task_root
    from proteus.bench.local import local_task
    from proteus.core import NEUTRAL
    from proteus.core.episode import RunConfig, run

    task = local_task("local:interval-merge")
    root = tmp_path / "r"
    cfg = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                    goal=as_goal(task), root=root, model="mock", episodes=1, seed=1,
                    task=task, grader_sandbox=trusted_grader)
    res = run(cfg)
    assert res.episodes_complete == 1
    assert task_root(root / "harness") == root / "task"
    assert (root / "task" / "solution.py").exists(), "task seeded outside the harness"
    assert not (root / "harness" / "task").exists()
    listed = subprocess.run(
        ["git", "--git-dir", str(root / ".snapshot.git"), "ls-tree", "-r",
         "--name-only", "HEAD"],
        capture_output=True, text=True, check=True).stdout
    assert "task/" not in listed, "the task workspace leaked into the snapshot"
    # and the evaluator still finds and grades it
    scored = res.eval_history[0]["results"]
    assert scored and scored[0]["name"].startswith("local:")
