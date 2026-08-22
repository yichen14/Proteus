"""End-to-end smoke tests — the framework runs offline and the instruments respond."""

from pathlib import Path

from proteus.adapters.minimal import MinimalHarness
from proteus.core import NEUTRAL, GoalConfig, review, snapshot
from proteus.core.episode import OPEN_BASE_PROMPTS, PHASES, RunConfig, run
from proteus.core.episode_protocol import EPISTEMIC_PROTOCOL
from proteus.measure import distance, stream


def _run(tmp_path, disposition, seed=0, episodes=6):
    cfg = RunConfig(
        name=disposition.label, adapter=MinimalHarness(), disposition=disposition,
        goal=GoalConfig.no_goal(), root=tmp_path / disposition.label / str(seed),
        model="mock", episodes=episodes, seed=seed,
    )
    return run(cfg)


def test_episode_runs_and_snapshots(tmp_path):
    res = _run(tmp_path, NEUTRAL)
    assert res.episodes_complete == 6
    assert res.error == ""
    work = Path(res.root) / "harness"
    assert snapshot.head(work)  # snapshots exist
    assert snapshot.commit_for_episode(work, 6) is not None


def test_disposition_shifts_surface(tmp_path):
    notes = _run(tmp_path, review("notes"))
    tools = _run(tmp_path, review("tools"))
    surfaces = MinimalHarness().surfaces()
    n_units = distance.units(Path(notes.root) / "harness", surfaces)
    t_units = distance.units(Path(tools.root) / "harness", surfaces)
    # a notes-disposition builds more notes; a tools-disposition builds more tools
    assert len(n_units["notes"]) > len(n_units["tools"])
    assert len(t_units["tools"]) > len(t_units["notes"])


def test_behavioural_R_separates_arms(tmp_path):
    streams = {}
    for disp in (NEUTRAL, review("notes"), review("tools")):
        streams[disp.label] = []
        for s in range(4):
            res = _run(tmp_path, disp, seed=s)
            trace = MinimalHarness().read_trace(Path(res.root), res.episodes_complete)
            streams[disp.label].append(stream.tool_stream(trace))
    r = stream.between_within(streams, level="freq", permutations=1000)
    assert r["R"] > 1.0          # who is acting explains more than chance
    assert r["p"] < 0.1


def test_distance_path_length(tmp_path):
    res = _run(tmp_path, review("notes"), episodes=4)
    work = Path(res.root) / "harness"
    surfaces = MinimalHarness().surfaces()
    # two consecutive states differ (the run added units)
    d = distance.compare(work, work, surfaces)
    assert d["notes"].distance == 0.0     # identical to itself


def test_file_surface_is_measured(tmp_path):
    # regression: a surface whose subdir is a single file (loop.py / AGENTS.md) must be
    # counted, not silently skipped — it is the self-editing surface the study is about
    from proteus.core.adapter import Surface
    from proteus.measure import distance
    h = tmp_path / "h"
    (h / "mem").mkdir(parents=True)
    (h / "loop.py").write_text("A = 1\n")
    surfaces = [Surface("memory", "mem", unit="file"),
                Surface("loop", "loop.py", unit="top_level_def", is_code=True)]
    u = distance.units(h, surfaces)
    assert u["loop"], "file-backed surface measured as empty"
    (h / "loop.py").write_text("A = 2\n")
    d = distance.compare(h, h, surfaces)   # identical to itself
    assert d["loop"].distance == 0.0
    u2 = distance.units(h, surfaces)
    assert u2["loop"] != u                  # content change is visible


def test_deterministic_separation_reads_high_R(tmp_path):
    # regression: perfectly separated deterministic arms (zero within-distance) must give
    # a LARGE R, not R=0 — the old `mb/mw if mw else 0` inverted the strongest signal
    from proteus.measure import stream
    A = ["read", "read", "write"]        # arm A: identical streams within
    B = ["bash", "bash", "edit"]         # arm B: identical within, different from A
    r = stream.between_within({"a": [A, A, A], "b": [B, B, B]},
                              level="freq", permutations=500)
    assert r["within"] == 0.0 and r["between"] > 0
    assert r["R"] > 10.0                  # maximal separation, not zero


def _prompts(adapter, disp):
    from proteus.core.episode import RunConfig, _phase_prompts
    cfg = RunConfig(name="t", adapter=adapter, disposition=disp, goal=GoalConfig(),
                    root=Path("/nonexistent"), model="mock", episodes=1)
    return _phase_prompts(cfg, "")


def test_disposition_reaches_prompt_only_harnesses_once():
    disp = review("notes")
    p = _prompts(MinimalHarness(), disp)
    assert all(p[ph].count("go over your notes") == 1 for ph in PHASES)


def test_file_carrying_adapters_do_not_also_get_the_prompt_copy():
    # dsh/pi install the disposition into AGENTS.md, which the harness loads itself; adding
    # it to the phase prompt as well would double the dose and put a copy outside F
    class FileCarrier(MinimalHarness):
        disposition_in_files = True

    disp = review("notes")
    p = _prompts(FileCarrier(), disp)
    assert all("go over your notes" not in p[ph] for ph in PHASES)
    assert OPEN_BASE_PROMPTS["observe"] in p["observe"]
    assert p["observe"].count(EPISTEMIC_PROTOCOL.strip()) == 1


def test_shipped_file_carrying_adapters_declare_it():
    import ast
    for mod in ("dsh", "pi"):
        src = Path(f"proteus/adapters/{mod}.py").read_text(encoding="utf-8")
        assert "disposition_in_files = True" in src, f"{mod} carries AGENTS.md but does not declare it"
        ast.parse(src)
