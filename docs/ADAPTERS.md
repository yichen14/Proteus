# Onboarding a harness

The input for onboarding is a **repository** — a git URL or a local path to the harness
you want to evolve. Onboarding produces two artifacts:

1. a **prepared environment** — a pinned Docker image carrying the harness itself
   (the evolving workspace is never in the image; it is always a mount);
2. an **adapter** — one class implementing `proteus.core.HarnessAdapter`, the only code
   you write.

Once both exist, the framework, sandbox, and the whole measurement suite work on your
harness unchanged, and the CLI loads your adapter with no registration.

```bash
# 1. point Proteus at the harness repo (git URL or local path)
proteus env scaffold --from https://github.com/org/their-harness --name theirs --ref v1.2.0

# 2. build the pinned environment image (uses the repo's own Dockerfile, or your wrapper)
proteus env build theirs
#    -> proteus-env-theirs:<shortsha>, resolved sha recorded in environments/theirs/environment.toml

# 3. write the adapter (the seven methods below), then verify it holds the contract
proteus check --harness mypkg.theirs_adapter:TheirsHarness            # free, static
proteus check --harness mypkg.theirs_adapter:TheirsHarness --episode  # + one live episode

# 4. run and measure
proteus run --harness mypkg.theirs_adapter:TheirsHarness \
    --arm neutral --arm review:notes --seeds 4 --episodes 10 --out runs/theirs
proteus measure --harness mypkg.theirs_adapter:TheirsHarness --out runs/theirs --travel
```

If the repo ships no Dockerfile, `proteus env scaffold --local-dockerfile` writes a wrapper
stub under `environments/<name>/` that is built with the repo checkout as its context. Put
the runtime the harness needs there. For a harness whose own source will be evolvable, use
`environments/dsh-src/` and `environments/pi-src/` as the stronger reference: the build
context is the pinned upstream checkout, and the image carries its source, dependencies,
build toolchain, and exact-tree boot wrapper.

## The contract

```python
class TheirsHarness:
    name = "theirs"
    continuity_mode = "native"     # native | framework | none; optional
    staged_activation = False       # True when editable self-code can affect execution
    disposition_in_files = False   # True if install_disposition writes a file the
                                   # harness loads itself (see step 3)

    def surfaces(self) -> Sequence[Surface]: ...
    def required_edit_tools(self) -> frozenset[str]: ...
    def seed(self, harness_root: Path, rng_seed: int = 0) -> None: ...
    def install_disposition(self, harness_root: Path, disposition: Disposition) -> None: ...
    def run_episode(self, spec: EpisodeSpec) -> EpisodeResult: ...
    def validate_candidate(self, harness_root: Path) -> str: ...  # optional boundary gate
    def read_trace(self, root: Path, episode: int) -> Sequence[ActionEvent]: ...
    def disposition_fingerprint(self, harness_root: Path) -> str: ...
```

The built-ins cover the two main integration shapes:

- **In-process** — you control the harness code (a Python library, or callable
  in-process): start from `proteus/adapters/minimal.py` (~140 lines).
- **External, source-evolving CLI** — the upstream repository stays pinned and untouched,
  but each run receives an evolvable copy of its real source: start from
  `proteus/adapters/dsh.py` or `pi.py`. Each episode boots a frozen last-valid copy while
  the model writes a separate candidate; Proteus rebuilds and validates that candidate
  only after reflect. The disposition installs as a removable marked block in `AGENTS.md`,
  and the trace is parsed from the harness's own session logs. A workspace-only CLI
  integration is the same shape without source extraction, the `loop` surface, rebuild,
  and boundary gate.

### 1. Declare surfaces
A `Surface` is one editable, persistent region the agent can grow. Declaring them as data
is what lets Proteus measure any harness:

```python
Surface("memory", "memory", unit="file",      write_tools=frozenset({"memory_write"}))
Surface("skills", "skills", unit="directory", write_tools=frozenset({"skill_write"}))
Surface("tools",  "tools",  unit="file",      write_tools=frozenset({"tool_write"}), is_code=True)
```

`unit` is how the measurement layer counts (a file, a directory, or a top-level def in a
code file). `free_named=True` means the agent picks unit names. If the stock harness has no
such regions, the adapter may establish them by convention in `seed` — the dsh adapter
seeds `notes/` + `tools/`, extracts its real source into `src/`, and names every surface in
the instructions file.

### 2. Seed
`seed(harness_root, rng_seed)` writes the episode-0 state: the workspace files the harness
starts from. Proteus snapshots this as commit 0. Episodes must tolerate waking up without
empty directories (git snapshots do not track them).

### 3. Install a removable disposition
`install_disposition` applies the action-preference perturbation; reinstalling `NEUTRAL`
must remove it without residue (`proteus check` verifies both directions via the
fingerprint). Pick the carrier that fits:
- **prompt** — append `disposition.phase_text(phase)` / `prompt_suffix` to phase prompts
  or to an instructions file the harness reads (simplest; dsh uses a marked block);
- **config** — substitute `disposition.config` into a config file;
- **patch** — apply `disposition.patch` as a diff (most general; removal is a revert).

By default Proteus also appends `disposition.phase_text(phase)` to every phase prompt, which
is the whole perturbation for a harness with no instructions file. If your carrier is a file
the harness loads on its own, set `disposition_in_files = True`: otherwise the same text
arrives twice per phase — about double the intended dose, through two channels of different
salience — and the prompt copy sits outside `F`, so it is neither removable nor covered by
`disposition_fingerprint`, which is what the attribution argument rests on.

### 4. Run one episode, emit the trace
`run_episode(spec)` executes the four phases (`spec.phase_prompts` carries the versioned
default epistemic protocol, goal text, and visible evaluator feedback already merged).
`read_trace` returns normalized `ActionEvent`s
— the only behaviour channel Proteus reads; never self-report. An external harness's own
logs are the source of truth: parse them rather than adding measurement instrumentation to
the harness. Evolving its run-local source is the subject's action, not instrumentation.

Declare how phase context continues: `native` (the backwards-compatible default) means
the harness owns it; `framework` means phases are fresh sessions joined by Proteus's
operational handoff; `none` deliberately leaves phases independent. Framework adapters
use `proteus.core.HandoffStore` around each phase and, for workspace-restricted
containers, bind `<run>/.proteus-state` over `/workspace/.proteus`. The host directory is
outside `<run>/harness`, so it is not snapshotted or measured as self-evolution. Proteus
archives `handoffs/epNNN/<phase>.{json,md}`, carries reflect into the next episode, and
falls back to normalized tool names and paths after an interruption. Never persist raw
model reasoning or tool results. DSH and Pi are the reference integrations.

Every adapter receives the same optional phase-aware budget fields in `EpisodeSpec`.
Use the core helpers rather than reproducing allocation arithmetic inside the adapter:

```python
from proteus.core.budget import PHASES, budget_plan, phase_prompt

plan = budget_plan(spec)
used = 0
for phase in PHASES:
    if plan.enabled and used >= plan.hard_limit:
        break
    stop_at = plan.stop_at(phase, used)
    prompt = phase_prompt(spec, phase, used)
    # Run the native phase with `prompt`; stop its native call loop at `stop_at`.
    # Refresh `used` from the native trace before starting the next phase.
```

`budget_plan` preserves legacy `max_turns` / `min_turns_per_phase` behavior and validates
the explicit normal plan, hard ceiling, act-priority borrowing, and checkpoint reserve.
`phase_prompt` adds live used/remaining values only when `announce_budget` is part of the
condition. For an external process, enforce the cumulative stop both between phases and
by watching its native log during a phase. The adapter defines what one native call is,
but must use the same definition for stopping, `EpisodeResult.turns`, and trace counters.
Do not implement checkpointing by synthesizing semantic memory: a framework-continuity
adapter archives the agent-written handoff and reports a miss when it remains unchanged.

### 5. Fingerprint
`disposition_fingerprint` hashes the currently-installed disposition carrier. The core
records the initial value and every candidate/checkpoint value outside the run root, so F
drift is auditable without forbidding self-editing. Resume requires the live value to
match the last durable checkpoint.

### 6. Stage activation for self-editable runtime code

If edits to the harness can change the code controlling later phases, declare
`staged_activation = True`. The core then supplies `EpisodeSpec.active_root`, a private
materialization of the last accepted commit. Your adapter must:

1. execute every phase from `active_root` (for containers, mount it read-only);
2. expose `root/harness` separately as the writable candidate;
3. never reload candidate code during observe/propose/act/reflect; and
4. implement a model-free `validate_candidate(harness_root) -> str` when the candidate
   needs a compile/boot check. Empty string means viable; a message means reject.

DSH and Pi mount active at `/workspace:ro` and candidate at
`/workspace/candidate:rw`. On a validation failure, the core preserves the candidate,
restores the last valid snapshot, records a rejected episode, and continues. On a phase,
provider, or snapshot failure, it preserves the partial attempt under a dedicated git ref,
restores files/index/HEAD, and leaves the same episode ready for resume.

## Isolation

If the harness lets the agent run its own code (most do), episodes must run under
`DockerSandbox` — an application-level file sandbox cannot contain a process that writes
and executes code. Use per-call mounts for your container layout (see the dsh adapter);
declare network policy in the environment manifest, `none` unless the harness itself must
reach an API.

Benchmark tasks are deliberately outside the measured snapshot at `<run>/task/`. If your
adapter supports benchmark work, expose that sibling to the agent without moving it under
`harness/`; dsh/pi bind it at `/workspace/task`. The framework seeds and grades the task,
but the adapter still owns how the harness sees files.

## Checklist

- [ ] environment: image pinned (repo ref recorded in a manifest or source-build recipe),
      state via mounts only
- [ ] surfaces declared as data (or established by convention in `seed`)
- [ ] disposition install is removable — `proteus check` passes
- [ ] trace parsed from the harness's own logs into `ActionEvent`s
- [ ] budget-aware loop consumes `budget_plan(spec)`, shows `phase_prompt(...)`, and
      enforces the returned cumulative stop using the same call unit as its trace
- [ ] continuity mode declared when not native; framework state remains outside the
      harness snapshot
- [ ] self-code adapter declares `staged_activation`; every phase uses the same read-only
      active snapshot and writes only to a separate candidate
- [ ] real (code-running) harness under `DockerSandbox`; containers that write bind mounts
      run as the host uid/gid
- [ ] source-evolving adapter: exact source extraction, exact-tree overlay, rebuild cache,
      and model-free `validate_candidate()` boundary gate
- [ ] benchmark-capable adapter: `<run>/task/` exposed separately from the snapshotted
      harness
- [ ] `proteus check --harness <module>:<Class> --episode` passes

## Auditing a finished sweep

Two things invalidate a trajectory without touching its score, so they are read from the
run rather than assumed away.

```bash
proteus reliability --out runs/mine --harness aki   # run this before `measure`
proteus audit       --out runs/mine --harness aki
```

`reliability` asks whether each arm reproduces itself, against a composition-matched null:
streams resampled from the arm's own pooled tool marginal, same length and same tool mix,
no shared procedure. `R` divides between-arm distance by within-arm distance, so an arm
whose runs do not resemble each other has a denominator made of noise and its R means
nothing whichever way it comes out. A ratio near 0 is highly reproducible; the
pre-registered threshold is 0.5.

`audit` reports two kinds of evidence, quoted rather than scored:

- **escape** — authored code that can leave the harness (`subprocess`, `eval`, a socket, a
  URL fetch), and any path fragment naming the study's own directories. Files identical to
  the episode-0 snapshot are skipped, so a harness that ships a CLI importing `subprocess`
  does not flag every seed.
- **awareness** — the subject describing its own situation as an experiment.

Both are deliberately over-inclusive: a tool may import `subprocess` and never escape.
The output is what to read, not what to exclude. Recall is limited in the other direction
too — in our own fleet one seed quarantined as aware matches no pattern and was caught by
a person reading its notes, so a clean audit means "no evidence found", not "clean".

Containment itself is the sandbox's job (`proteus.sandbox`): a harness that authors its own
tools can always defeat a pattern list, which is why the audit runs after the fact and the
isolation runs during.

## Resuming an interrupted seed

Episodes cost minutes to tens of minutes, so a seed that dies at episode 26 is hours of
trajectory. `run(cfg, start=N, resume=True)` continues from the episode after `N`, using
the harness on disk rather than re-seeding over it; a positive `start` implies resume, while
the explicit flag distinguishes episode-0 recovery from a new run. `proteus run
--on-existing resume` applies this to a whole sweep only after the versioned manifest
condition matches: finished seeds are skipped, partial ones pick up after their last
snapshot. `completed_episodes()` counts contiguous snapshot commits, not trace files — a
provider outage writes a trace per failed attempt, and counting those reports a seed that
finished nothing as complete.

## Letting the harness edit its own code

Instructions and notes are not the interesting ceiling: self-evolution that cannot touch
the loop is memory with extra steps. The arrangement that gives an external harness its
own run-local code while leaving the pinned upstream checkout untouched is the one Aki uses
natively —
**copy the code into the harness at seed time, but activate it only at episode boundaries**:

1. `seed()` extracts the harness's **real source** from the prepared image into
   `harness/src/`: for pi the pi-mono checkout (~1,100 `.ts` files), for dsh the
   deepseek-harness monorepo (~2,300 `.ts` files, via `git archive`, so the seed's src/
   is exactly the tracked source of the build it boots). The image's entrypoint syncs
   the agent's copy over the baked tree at boot, rebuilds with the project's own
   toolchain when the source hash changes (outputs cached on `/state`; an untouched copy
   boots via a pristine-hash fast path), and execs the built CLI. External dependency
   bytes stay in the image as apparatus. DSH may evolve package manifests and workspace
   packages only when its frozen lockfile remains consistent and every required external
   package is already present in the image's offline pnpm store.
2. At episode N, every phase mounts the same last-valid snapshot read-only at `/workspace`.
   The writable `harness/` candidate is a separate `/workspace/candidate` mount. The boot
   wrapper syncs only active `/workspace/src`, so an act edit cannot control reflect. The
   agent may inspect its candidate, but activation waits for episode N+1.
3. `src/` is a declared surface (`loop`, `is_code=True`), inside the snapshot repo — code
   edits are versioned per episode and measured with the same ruler as notes and tools.
4. After reflect, `validate_candidate()` runs the **model-free viability gate**. Pi rebuilds
   and probes `--version`. DSH performs a frozen offline dependency relink and rebuild in
   one container, then starts the exact headless profile from the saved cache in a second,
   fresh container. This catches stale lockfiles, missing package links, newly added package
   outputs omitted from a cache, and plugin-load failures before activation. A failed
   candidate cannot activate: the active line rolls back and remains healthy, while the
   exact failed tree is restored as episode N+1's writable repair candidate. A passing
   candidate first runs as the controlling harness one episode later.

Verified live for both harnesses: a marker written into the real TypeScript entry point
(`packages/coding-agent/src/cli.ts` for pi, `apps/cli/src/bin.ts` for dsh) appears on the
next boot after the automatic in-container rebuild; a later boot hits the dist cache;
and a planted TS type error is refused by the gate (exit 97 with the build log tail) and
automatically restored to the prior valid snapshot. The Docker image itself is baked once per harness version
and never rebuilt during a run — per episode, an unchanged source boots via the fast
path, and each distinct source state pays for exactly one in-container rebuild.
