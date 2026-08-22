"""The self-code arrangement for containerized harnesses, offline.

Both dsh and pi run source mode: seed() unpacks the image's source tar, every episode runs
a frozen snapshot at /workspace, and the boundary gate rebuilds /workspace/candidate. The live
proofs (real extraction, a marker edit surviving the rebuild, a planted error refused)
need Docker and run in the release smoke; these cover the adapter logic with a fake
sandbox.
"""

import shutil
import subprocess
from pathlib import Path

from proteus.adapters.dsh import DshHarness
from proteus.adapters.pi import PiHarness


GATE_COMMANDS = {("--version",), ("--proteus-headless-smoke",)}


def _is_gate(call):
    return tuple(call["command"]) in GATE_COMMANDS


class FakeSandbox:
    def __init__(self, boot_rc=0, cold_rc=0):
        self.boot_rc = boot_rc
        self.cold_rc = cold_rc
        self.calls = []

    def run(self, run_root, command, env, timeout_s, mounts=(), stop_check=None):
        self.calls.append({"command": command, "mounts": mounts,
                           "stop_check": stop_check, "env": env})
        if tuple(command) not in GATE_COMMANDS and stop_check is not None and stop_check():
            return subprocess.CompletedProcess(command, 137, "", "killed")
        if command == ["--version"]:
            rc = self.boot_rc
        elif command == ["--proteus-headless-smoke"]:
            rc = self.cold_rc
        else:
            rc = 0
        return subprocess.CompletedProcess(command, rc, "ok", "boom" if rc else "")


def _seed_with_fake_src(adapter, harness: Path, pieces):
    adapter._extract_self_code = lambda dest: None      # no docker in offline tests
    adapter.seed(harness, 0)
    for piece in pieces:
        p = harness / "src" / piece
        if "." in piece:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        else:
            p.mkdir(parents=True, exist_ok=True)
            (p / "code.js").write_text("// self\n")


def test_loop_surface_is_declared_and_mapped():
    for cls in (DshHarness, PiHarness):
        a = cls(key="x", sandbox=FakeSandbox())
        names = {s.name: s for s in a.surfaces()}
        assert "loop" in names and names["loop"].is_code
        assert a._surface_for_path("/workspace/src/lib/bin.js") == "loop"
        assert a._surface_for_path("/workspace/candidate/src/lib/bin.js") == "loop"


def test_source_evolving_adapters_stage_activation():
    for cls in (DshHarness, PiHarness):
        adapter = cls(key="x", sandbox=FakeSandbox())
        assert adapter.staged_activation
        assert callable(adapter.validate_candidate)


def test_instruction_carrier_handles_per_phase_only_and_neutral_cleanup(tmp_path):
    from proteus.adapters import instructions
    from proteus.core.disposition import Disposition, NEUTRAL

    path = tmp_path / "AGENTS.md"
    path.write_text("# Base\n")
    instructions.install_block(
        path, Disposition(label="phase", per_phase={"act": "Prefer small edits."}))
    assert "During act: Prefer small edits." in path.read_text()
    instructions.install_block(path, NEUTRAL)
    assert path.read_text() == "# Base\n"


def test_pi_live_budget_recognizes_all_tool_call_markers(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    session = state / "s.jsonl"
    session.write_text('\"toolCall\"\n\"tool_call\"\n\"toolUse\"\n')
    adapter = PiHarness(key="x", sandbox=FakeSandbox())
    assert adapter._live_calls(state, {session}, set()) == 3


def test_read_trace_aggregates_multiple_sessions_per_phase(tmp_path):
    import json

    root = tmp_path / "run"
    state = root / ".pi-state"
    traces = root / "traces"
    state.mkdir(parents=True)
    traces.mkdir()

    def event(name):
        return json.dumps({
            "type": "message", "message": {"role": "assistant", "content": [
                {"type": "toolCall", "name": name, "arguments": {}}]}})

    (state / "a.jsonl").write_text(event("first") + "\n")
    (state / "b.jsonl").write_text(event("second") + "\n")
    (traces / "ep001.json").write_text(json.dumps({"act": ["a.jsonl", "b.jsonl"]}))
    trace = PiHarness(key="x", sandbox=FakeSandbox()).read_trace(root, 1)
    assert [item.tool for item in trace] == ["first", "second"]
    assert [item.turn for item in trace] == [1, 2]


def test_source_mode_gates_through_the_boot_contract(tmp_path):
    # the boot wrapper rebuilds from /workspace/src, so the gate needs no extra mounts:
    # workspace + state are the whole contract, for both containerized harnesses
    from proteus.core.adapter import EpisodeSpec
    for i, cls in enumerate((DshHarness, PiHarness)):
        sandbox = FakeSandbox()
        a = cls(key="x", sandbox=sandbox)
        h = tmp_path / f"harness{i}"
        _seed_with_fake_src(a, h, ("packages",))
        root = h.parent / f"root{i}"
        root.mkdir(exist_ok=True)
        (root / "harness").symlink_to(h)
        a.run_episode(EpisodeSpec(root=root, episode=1, model="m", phase_prompts={}))
        gates = [call for call in sandbox.calls if _is_gate(call)]
        expected = (["--version"], ["--proteus-headless-smoke"]) \
            if cls is DshHarness else (["--version"],)
        assert [call["command"] for call in gates] == list(expected), cls.__name__
        for gate in gates:
            conts = {cont for _, cont in gate["mounts"]}
            assert conts == {"/workspace", "/state"}, \
                f"{cls.__name__}: the gate must run exactly the boot contract"


def test_broken_self_code_fails_the_episode_legibly(tmp_path):
    from proteus.core.adapter import EpisodeSpec
    for i, cls in enumerate((DshHarness, PiHarness)):
        sandbox = FakeSandbox(boot_rc=97)
        a = cls(key="x", sandbox=sandbox)
        h = tmp_path / f"harness{i}"
        _seed_with_fake_src(a, h, ("packages",))
        root = h.parent / f"root{i}"
        root.mkdir(exist_ok=True)
        (root / "harness").symlink_to(h)
        res = a.run_episode(EpisodeSpec(root=root, episode=1, model="m", phase_prompts={}))
        assert not res.ok and "does not boot" in res.error, cls.__name__
        # the gate ran once and no phase was attempted after it
        assert [c["command"] for c in sandbox.calls] == [["--version"]], cls.__name__


def test_dsh_headless_cold_start_failure_is_a_viability_error(tmp_path):
    sandbox = FakeSandbox(cold_rc=98)
    adapter = DshHarness(key="x", sandbox=sandbox)
    harness = tmp_path / "harness"
    _seed_with_fake_src(adapter, harness, ("packages",))

    error = adapter.check_boot(harness)

    assert "fails headless cold start" in error
    assert [call["command"] for call in sandbox.calls] == [
        ["--version"], ["--proteus-headless-smoke"]]


def test_dsh_permission_mode_reaches_every_phase(tmp_path):
    from proteus.core.adapter import EpisodeSpec

    sandbox = FakeSandbox()
    adapter = DshHarness(
        key="x", sandbox=sandbox, permission_mode="danger-full-access"
    )
    harness = tmp_path / "harness"
    _seed_with_fake_src(adapter, harness, ("packages",))
    root = tmp_path / "root"
    root.mkdir()
    (root / "harness").symlink_to(harness)
    adapter.run_episode(EpisodeSpec(
        root=root,
        episode=1,
        model="m",
        phase_prompts={phase: phase for phase in ("observe", "propose", "act", "reflect")},
    ))

    phase_calls = [call for call in sandbox.calls if not _is_gate(call)]
    assert len(phase_calls) == 4
    assert all(call["env"]["DSH_PERMISSION_MODE"] == "danger-full-access"
               for call in phase_calls)


def test_framework_handoff_mount_is_writable_but_snapshot_external(tmp_path):
    from proteus.core.adapter import EpisodeSpec

    for i, cls in enumerate((DshHarness, PiHarness)):
        sandbox = FakeSandbox()
        adapter = cls(key="x", sandbox=sandbox)
        harness = tmp_path / f"handoff-harness-{i}"
        _seed_with_fake_src(adapter, harness, ())
        root = tmp_path / f"handoff-root-{i}"
        root.mkdir()
        (root / "harness").symlink_to(harness)
        adapter.run_episode(EpisodeSpec(root=root, episode=1, model="m", phase_prompts={}))

        phase = next(call for call in sandbox.calls if not _is_gate(call))
        mounts = dict(phase["mounts"])
        assert mounts[str(root / ".proteus-state")] == "/workspace/.proteus"
        assert not str(root / ".proteus-state").startswith(str(harness))


def test_staged_episode_mounts_frozen_active_read_only_and_candidate_writable(tmp_path):
    from proteus.core.adapter import EpisodeSpec

    for i, cls in enumerate((DshHarness, PiHarness)):
        sandbox = FakeSandbox()
        adapter = cls(key="x", sandbox=sandbox)
        root = tmp_path / f"staged-root-{i}"
        harness = root / "harness"
        active = tmp_path / f"private-active-{i}"
        root.mkdir()
        _seed_with_fake_src(adapter, harness, ())
        shutil.copytree(harness, active)

        adapter.run_episode(EpisodeSpec(
            root=root, episode=1, model="m", phase_prompts={}, active_root=active,
        ))

        phase_calls = [call for call in sandbox.calls if not _is_gate(call)]
        assert len(phase_calls) == 4
        assert not [call for call in sandbox.calls if _is_gate(call)], \
            "a staged episode must not preflight the writable candidate before its phases"
        assert (active / "candidate").is_dir()
        assert (active / ".proteus").is_dir()
        for call in phase_calls:
            mounts = call["mounts"]
            assert (str(active), "/workspace", "ro") in mounts
            assert (str(harness), "/workspace/candidate") in mounts
            assert (str(root / ".proteus-state"), "/workspace/.proteus") in mounts


def test_staged_prompt_forbids_same_episode_candidate_activation(tmp_path):
    from proteus.core import GoalConfig, NEUTRAL
    from proteus.core.episode import PHASES, RunConfig, _phase_prompts

    adapter = DshHarness(key="x", sandbox=FakeSandbox())
    cfg = RunConfig(name="t", adapter=adapter, disposition=NEUTRAL,
                    goal=GoalConfig(), root=tmp_path, model="mock")
    prompts = _phase_prompts(cfg, "")
    for phase in PHASES:
        assert "/workspace/candidate" in prompts[phase]
        assert "including reflect" in prompts[phase]
        assert "next episode" in prompts[phase]
        assert "Proteus epistemic protocol" in prompts[phase]


def test_reseeding_never_overwrites_evolved_code(tmp_path):
    a = DshHarness(key="x", sandbox=FakeSandbox())
    h = tmp_path / "harness"
    _seed_with_fake_src(a, h, ("lib",))
    (h / "src" / "lib" / "code.js").write_text("// evolved by the agent\n")
    calls = []
    a._extract_self_code = lambda dest: calls.append(dest)

    real = DshHarness(key="x", sandbox=FakeSandbox())
    # the real guard lives in _extract_self_code itself: non-empty src is left alone
    real._extract_self_code(h / "src")
    assert (h / "src" / "lib" / "code.js").read_text() == "// evolved by the agent\n"


def test_source_hash_frames_file_boundaries_and_symlinks(tmp_path):
    """Different exact trees must never share a build-cache identity."""
    if shutil.which("node") is None:
        return  # the runtime images always carry Node; keep the Python-only runner usable

    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "one").write_text("ab")
    (left / "two").write_text("c")
    (right / "one").write_text("a")
    (right / "two").write_text("bc")

    scripts = (Path("environments/dsh-src/boot.sh"),
               Path("environments/pi-src/boot.sh"))

    def digest(script, root):
        return subprocess.run(
            ["sh", str(script), "--proteus-tree-hash", str(root)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    for script in scripts:
        assert digest(script, left) != digest(script, right), \
            f"{script}: file-boundary edit collided"
        clean = digest(script, left)
        (left / "node_modules").mkdir()
        (left / "node_modules" / "untrusted.js").write_text("shadow baked dependencies")
        assert digest(script, left) == clean, f"{script}: node_modules entered source hash"
        shutil.rmtree(left / "node_modules")
        regular = digest(script, left)
        (left / "one").chmod(0o744)
        assert digest(script, left) != regular, f"{script}: executable bit was not hashed"
        (left / "one").chmod(0o644)
        link = left / "link"
        link.unlink(missing_ok=True)
        link.symlink_to("one")
        before = digest(script, left)
        link.unlink()
        link.symlink_to("two")
        assert digest(script, left) != before, f"{script}: symlink target was not hashed"


def test_dsh_boot_contract_relinks_dependencies_and_caches_new_packages():
    script = Path("environments/dsh-src/boot.sh").read_text()
    dockerfile = Path("environments/dsh-src/Dockerfile").read_text()

    assert "pnpm install --offline --frozen-lockfile" in script
    assert "--config.trust-lockfile=true" in script
    assert '--store-dir "$PNPM_STORE"' in script
    assert 'DEP_HASH=$(tree_hash /workspace/src dependencies)' in script
    assert "pristine-dependency-hash" in script
    assert "COREPACK_ENABLE_NETWORK=0" in script
    assert "find $DISTS" not in script
    assert "find apps packages vendor" in script
    assert "Native launcher artifacts are baked" in script
    assert "-path '*/lib/*'" in script
    assert 'CACHE_TMP="/state/.dist-$HASH.$$.tar"' in script
    assert '"--proteus-headless-smoke"' in script
    assert "MISSING_CREDENTIAL" in script
    assert "pnpm install --frozen-lockfile --store-dir /opt/pnpm-store" in dockerfile
    assert "COREPACK_HOME=/opt/corepack" in dockerfile
    assert "COREPACK_ENABLE_NETWORK=0" in dockerfile
    assert "--proteus-dependency-hash" in dockerfile
    assert "chmod -R a+rwX /opt/src /opt/corepack /opt/pnpm-store" in dockerfile


def test_dsh_dependency_hash_ignores_code_but_tracks_package_inputs(tmp_path):
    script = Path("environments/dsh-src/boot.sh")
    source = tmp_path / "source"
    package = source / "packages" / "audio"
    patches = source / "patches"
    package.mkdir(parents=True)
    patches.mkdir()
    (source / "package.json").write_text('{"packageManager":"pnpm@11.7.0"}')
    (source / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")
    (source / "pnpm-workspace.yaml").write_text("packages: ['packages/*']\n")
    (package / "package.json").write_text('{"name":"audio"}')
    code = package / "index.ts"
    code.write_text("export const version = 1;\n")
    patch = patches / "dependency.patch"
    patch.write_text("first\n")

    def digest():
        return subprocess.run(
            ["sh", str(script), "--proteus-dependency-hash", str(source)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    baseline = digest()
    code.write_text("export const version = 2;\n")
    assert digest() == baseline
    (package / "package.json").write_text('{"name":"audio","dependencies":{"x":"1"}}')
    package_changed = digest()
    assert package_changed != baseline
    patch.write_text("second\n")
    assert digest() != package_changed


# ------------------------------------------------------------------- turn budget

def test_explicit_budget_validation_is_strict():
    from proteus.core.budget import make_budget_plan

    base = {"observe": 2, "propose": 2, "act": 6, "reflect": 2}
    invalid = [
        ({"phase_turns": {**base, "act": 5}}, "sum to 11"),
        ({"phase_turns": base, "hard_max_turns": 11}, "must be at least"),
        ({"phase_turns": base, "min_turns_per_phase": 1}, "cannot be combined"),
        ({"phase_turns": base, "checkpoint_turns": 3}, "smallest"),
    ]
    for overrides, expected in invalid:
        try:
            make_budget_plan(max_turns=12, **overrides)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"invalid budget was accepted: {overrides}")

def test_budget_stops_new_phases_exactly(tmp_path):
    # the between-phase check is exact and needs nothing from the log format
    from proteus.core.adapter import EpisodeSpec
    for i, cls in enumerate((DshHarness, PiHarness)):
        sandbox = FakeSandbox()
        a = cls(key="x", sandbox=sandbox)
        a._live_calls = lambda *args, **kw: 99          # already over budget
        h = tmp_path / f"h{i}"
        _seed_with_fake_src(a, h, ())
        root = tmp_path / f"r{i}"
        root.mkdir()
        (root / "harness").symlink_to(h)
        res = a.run_episode(EpisodeSpec(root=root, episode=1, model="m",
                                        phase_prompts={}, max_turns=10))
        assert res.ok and not res.error, cls.__name__
        assert res.counters["turn_capped"], cls.__name__
        phases = [c for c in sandbox.calls if not _is_gate(c)]
        assert phases == [], f"{cls.__name__}: a phase ran past the budget"


def test_budget_kill_mid_phase_is_a_cap_not_an_error(tmp_path):
    from proteus.core.adapter import EpisodeSpec
    for i, cls in enumerate((DshHarness, PiHarness)):
        sandbox = FakeSandbox()
        a = cls(key="x", sandbox=sandbox)
        calls = iter([0, 50])                            # under budget at launch, over mid-phase
        a._live_calls = lambda *args, **kw: next(calls, 50)
        h = tmp_path / f"h{i}"
        _seed_with_fake_src(a, h, ())
        root = tmp_path / f"r{i}"
        root.mkdir()
        (root / "harness").symlink_to(h)
        res = a.run_episode(EpisodeSpec(root=root, episode=1, model="m",
                                        phase_prompts={}, max_turns=10))
        assert res.ok and not res.error, \
            f"{cls.__name__}: a budget kill must be a cap, got error={res.error!r}"
        assert res.counters["turn_capped"], cls.__name__
        phases = [c for c in sandbox.calls if not _is_gate(c)]
        assert len(phases) == 1, f"{cls.__name__}: phases continued after the kill"


def test_no_budget_means_no_watching(tmp_path):
    from proteus.core.adapter import EpisodeSpec
    sandbox = FakeSandbox()
    a = PiHarness(key="x", sandbox=sandbox)
    h = tmp_path / "h"
    _seed_with_fake_src(a, h, ())
    root = tmp_path / "r"
    root.mkdir()
    (root / "harness").symlink_to(h)
    a.run_episode(EpisodeSpec(root=root, episode=1, model="m", phase_prompts={},
                              max_turns=0))
    phases = [c for c in sandbox.calls if not _is_gate(c)]
    assert phases and all(c["stop_check"] is None for c in phases)


def test_announced_budget_reaches_every_phase_prompt(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import PHASES, RunConfig, _phase_prompts
    on = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                   goal=GoalConfig(), root=tmp_path, model="mock", max_turns=12,
                   announce_budget=True)
    p = _phase_prompts(on, "")
    assert all("at most 12 tool calls" in p[ph] for ph in PHASES)
    off = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                    goal=GoalConfig(), root=tmp_path, model="mock", max_turns=12)
    q = _phase_prompts(off, "")
    assert all("tool calls in this episode" not in q[ph] for ph in PHASES)


def test_context_fresh_phase_prompts_require_file_handoffs(tmp_path):
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import PHASES, RunConfig, _phase_prompts

    class FreshAdapter:
        continuity_mode = "framework"
        disposition_in_files = False

    cfg = RunConfig(name="t", adapter=FreshAdapter(), disposition=NEUTRAL,
                    goal=GoalConfig(), root=tmp_path, model="mock")
    prompts = _phase_prompts(cfg, "")
    assert all("/workspace/.proteus/handoff.md" in prompts[phase] for phase in PHASES)
    assert all("raw tool output" in prompts[phase] for phase in PHASES)


def test_non_framework_harnesses_do_not_receive_file_protocol(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import PHASES, RunConfig, _phase_prompts

    cfg = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                    goal=GoalConfig(), root=tmp_path, model="mock")
    prompts = _phase_prompts(cfg, "")
    assert all("/workspace/.proteus" not in prompts[phase] for phase in PHASES)


# ------------------------------------------------------------- per-phase reservation

def test_reservation_forces_every_phase_to_run(tmp_path):
    # a policy that would spend the whole budget in observe must be cut at the stop
    # line so propose/act/reflect still get their reserved turns
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import RunConfig, run

    def greedy(phase, prompt, episode, rng):
        return [("read_state", None, f"{phase}-{i}") for i in range(50)]

    adapter = MinimalHarness(policy=greedy)
    cfg = RunConfig(name="t", adapter=adapter, disposition=NEUTRAL, goal=GoalConfig(),
                    root=tmp_path / "r", model="mock", episodes=1, seed=1,
                    max_turns=8, min_turns_per_phase=2)
    res = run(cfg)
    trace = adapter.read_trace(tmp_path / "r", 1)
    by_phase = {}
    for e in trace:
        by_phase[e.phase] = by_phase.get(e.phase, 0) + 1
    assert set(by_phase) == {"observe", "propose", "act", "reflect"}, \
        f"a phase starved: {by_phase}"
    assert by_phase["observe"] == 2, f"observe ran past its stop line: {by_phase}"
    assert sum(by_phase.values()) <= 8
    assert res.episodes_complete == 1


def test_reservation_stop_is_not_an_episode_end_for_containers(tmp_path):
    # a mid-phase kill at the reservation line must move to the next phase, and only a
    # spent budget ends the episode
    from proteus.core.adapter import EpisodeSpec
    for i, cls in enumerate((DshHarness, PiHarness)):
        sandbox = FakeSandbox()
        a = cls(key="x", sandbox=sandbox)
        # scripted counts: launch checks say under budget; each phase's mid-phase read
        # sits exactly on its reservation line until the last, which spends the budget
        # three reads per fired phase: launch check, mid-phase stop, budget re-check
        script = iter([0, 2, 2,      # phase 1 (stop_at=8-6=2), re-check under budget
                       2, 4, 4,      # phase 2 (stop_at=4)
                       4, 6, 6,      # phase 3 (stop_at=6)
                       6, 8, 8])     # phase 4 (stop_at=8), re-check spends the budget
        a._live_calls = lambda *args, **kw: next(script, 8)
        h = tmp_path / f"h{i}"
        _seed_with_fake_src(a, h, ())
        root = tmp_path / f"r{i}"
        root.mkdir()
        (root / "harness").symlink_to(h)
        res = a.run_episode(EpisodeSpec(root=root, episode=1, model="m", phase_prompts={},
                                        max_turns=8, min_turns_per_phase=2))
        phases = [c for c in sandbox.calls if not _is_gate(c)]
        assert len(phases) == 4, \
            f"{cls.__name__}: reservation stop ended the episode after {len(phases)} phases"
        assert res.ok and res.counters["turn_capped"], cls.__name__


def test_budget_must_cover_the_reserves(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import RunConfig, run
    cfg = RunConfig(name="t", adapter=MinimalHarness(), disposition=NEUTRAL,
                    goal=GoalConfig(), root=tmp_path / "r", model="mock", episodes=1,
                    max_turns=7, min_turns_per_phase=2)
    try:
        run(cfg)
    except ValueError as exc:
        assert "min_turns_per_phase" in str(exc)
    else:
        raise AssertionError("a budget below 4x the reserve must be refused")


def test_explicit_phase_budget_gives_unused_and_burst_capacity_to_act(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core import NEUTRAL, GoalConfig
    from proteus.core.episode import RunConfig, run

    def policy(phase, prompt, episode, rng):
        count = 1 if phase in ("observe", "propose") else 50
        return [("read_state", None, f"{phase}-{i}") for i in range(count)]

    adapter = MinimalHarness(policy=policy)
    result = run(RunConfig(
        name="budget", adapter=adapter, disposition=NEUTRAL, goal=GoalConfig(),
        root=tmp_path / "explicit", model="mock", episodes=1,
        max_turns=12, hard_max_turns=20,
        phase_turns={"observe": 2, "propose": 2, "act": 6, "reflect": 2},
    ))
    trace = adapter.read_trace(tmp_path / "explicit", 1)
    counts = {phase: sum(event.phase == phase for event in trace)
              for phase in ("observe", "propose", "act", "reflect")}
    assert counts == {"observe": 1, "propose": 1, "act": 16, "reflect": 2}
    assert result.eval_history[0]["counters"]["phase_act_turns"] == 16


def test_explicit_phase_budget_does_not_let_reflect_borrow_act_capacity(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core.adapter import EpisodeSpec

    def policy(phase, prompt, episode, rng):
        count = 50 if phase == "reflect" else 1
        return [("read_state", None, f"{phase}-{i}") for i in range(count)]

    adapter = MinimalHarness(policy=policy)
    root = tmp_path / "reflect-cap"
    adapter.seed(root / "harness")
    result = adapter.run_episode(EpisodeSpec(
        root=root, episode=1, model="mock", phase_prompts={},
        max_turns=12, hard_max_turns=20,
        phase_turns={"observe": 2, "propose": 2, "act": 6, "reflect": 2},
    ))
    assert result.counters["phase_reflect_turns"] == 2
    assert result.turns == 5


def test_live_budget_prompt_reports_phase_start_state(tmp_path):
    from proteus.adapters.minimal import MinimalHarness
    from proteus.core.adapter import EpisodeSpec

    seen = {}

    def policy(phase, prompt, episode, rng):
        seen[phase] = prompt
        return [("read_state", None, phase)]

    adapter = MinimalHarness(policy=policy)
    root = tmp_path / "live-prompt"
    adapter.seed(root / "harness")
    result = adapter.run_episode(EpisodeSpec(
        root=root, episode=7, model="mock", phase_prompts={}, announce_budget=True,
        max_turns=8, hard_max_turns=12,
        phase_turns={"observe": 2, "propose": 2, "act": 2, "reflect": 2},
    ))
    assert result.ok
    assert "episode 7, phase observe" in seen["observe"]
    assert "calls already used before this phase: 0" in seen["observe"]
    assert "calls already used before this phase: 2" in seen["act"]
    assert "episode hard ceiling: 12" in seen["act"]
    assert "act owns unused earlier-phase allowance and up to 4 burst calls" in seen["act"]


def test_checkpoint_reserve_is_agent_authored_and_visible():
    from proteus.core.adapter import EpisodeSpec
    from proteus.core.budget import phase_prompt

    spec = EpisodeSpec(
        root=Path("."), episode=3, model="m", phase_prompts={"act": "do work"},
        max_turns=12, hard_max_turns=20, checkpoint_turns=2,
        phase_turns={"observe": 2, "propose": 2, "act": 6, "reflect": 2},
        announce_budget=True, continuity_mode="framework",
    )
    prompt = phase_prompt(spec, "act", 4)
    assert "checkpoint reserve: keep final 2 calls" in prompt
    assert "cumulative call 16" in prompt
    assert "/workspace/.proteus/handoff.md" in prompt
    assert "will not invent a semantic summary" in prompt


def test_explicit_phase_budget_stops_container_phases_at_the_same_lines(tmp_path):
    from proteus.core.adapter import EpisodeSpec

    for i, cls in enumerate((DshHarness, PiHarness)):
        sandbox = FakeSandbox()
        adapter = cls(key="x", sandbox=sandbox)
        script = iter([0, 2, 2, 2, 4, 4, 4, 18, 18, 18, 20, 20])
        adapter._live_calls = lambda *args, **kw: next(script, 20)
        harness = tmp_path / f"explicit-container-harness-{i}"
        _seed_with_fake_src(adapter, harness, ())
        root = tmp_path / f"explicit-container-root-{i}"
        root.mkdir()
        (root / "harness").symlink_to(harness)
        result = adapter.run_episode(EpisodeSpec(
            root=root, episode=1, model="m", phase_prompts={},
            max_turns=12, hard_max_turns=20, checkpoint_turns=1,
            phase_turns={"observe": 2, "propose": 2, "act": 6, "reflect": 2},
            announce_budget=True, continuity_mode="framework",
        ))
        phases = [call for call in sandbox.calls if not _is_gate(call)]
        assert len(phases) == 4, cls.__name__
        assert result.ok and result.counters["turn_capped"], cls.__name__
        assert result.counters["checkpoint_misses"] == 4, cls.__name__
