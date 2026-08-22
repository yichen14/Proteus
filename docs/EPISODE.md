# The episode: what updates when, and who owns it

Code: [`proteus/core/episode.py`](../proteus/core/episode.py) (the loop),
[`proteus/core/budget.py`](../proteus/core/budget.py) (budget protocol),
[`proteus/core/continuity.py`](../proteus/core/continuity.py) (phase handoffs),
[`proteus/core/goal.py`](../proteus/core/goal.py) (goals and evaluators),
[`proteus/core/snapshot.py`](../proteus/core/snapshot.py) (snapshots),
[`proteus/core/adapter.py`](../proteus/core/adapter.py) (the adapter contract).

The whole division of labour is one sentence: **the framework owns the transaction; the
adapter owns harness execution.** Proteus owns snapshots, candidate acceptance, rollback,
records, and resume. The adapter owns how a particular harness runs a phase and how its
candidate is validated. This division is what makes a no-goal Aki run and a
goal-conditioned pi run readable with the same ruler without pretending their runtimes
are identical.

---

## At run start (once per seed, not per episode)

| step | owner | what happens |
|---|---|---|
| 1. `adapter.seed(harness_root, rng_seed)` | adapter | lay down the episode-0 state: surface directories, initial instructions; dsh/pi also extract their own code from the image into `harness/src/` |
| 2. `adapter.install_disposition(...)` | adapter | install the action-preference perturbation — must be removable; the carrier is the adapter's choice (§ contract) |
| 3. task seeding (benchmark runs only) | framework | `seed_task` writes the benchmark task into `<run>/task/`, beside the harness and outside its snapshot; dsh/pi mount it at `/workspace/task` |
| 4. `snapshot.init(harness)` | framework | bare git repo; **every ignore rule disabled** (the harness is the measured object — nothing in it may be invisible to the instrument); commit `episode 0` |

Resume (`run(cfg, start=N, resume=True)`) skips all four and continues on the evolved
harness on disk from episode N+1. Positive `start` values imply resume for API
compatibility; the explicit flag matters at N=0, where a new run and a crashed-before-
episode-1 run otherwise look identical. `completed_episodes` counts **contiguous snapshot
commits**, not trace files — a provider outage writes a trace per failed attempt. Before
any resume, Proteus restores files, index, and HEAD to the exact episode-N checkpoint
(including episode 0), reconciles the framework handoff, and verifies private records.
This removes a half-written candidate left by SIGKILL or a machine restart before it can
leak into the next attempt.

## Every episode N

```
assemble prompts → materialize last-valid active snapshot → run four phases against it
while writing a separate candidate → read trace → boundary viability gate → evaluators
→ selection → promote or preserve+restore → records & feedback → episode N+1 activates
```

## Scope of the transaction contract

The current implementation has two layers that must not be conflated:

| layer | applies to | framework guarantee |
|---|---|---|
| snapshot transaction and recovery | every adapter | accepted/rejected history, automatic restore after adapter or snapshot failure, strict resume from the last complete checkpoint |
| frozen active + writable candidate | adapters declaring `staged_activation=True` | private `active_root`, isolation prompt, episode-boundary activation |
| candidate viability | adapters implementing `validate_candidate()` | model-free gate before evaluators; failure cannot activate, and staged adapters retain its exact tree as the next writable repair base |

DSH and Pi implement all three layers. Minimal and LLM do not execute an editable copy of
their own runtime, so they currently use the common snapshot transaction without staged
activation. Aki delegates episode execution to its native supervisor. A custom adapter
that omits `staged_activation` receives no `active_root`; Proteus therefore assumes that
adapter already owns any required episode-atomic execution semantics.

This is currently an **explicit capability contract, not automatic enforcement for an
arbitrary third-party adapter**. The framework can provide a frozen tree, but it cannot
prove that an opaque external process actually executed it. A self-code adapter is
conformant only if it declares staged activation, runs every phase from `active_root`,
writes only to `root/harness`, and supplies a model-free validator when its candidate can
fail to build or boot.

### 1. Assemble the four phase prompts — framework

`_phase_prompts` starts from the four base texts (observe / propose / act / reflect) and
applies fixed rules:

- the versioned **default epistemic protocol** joins all four phases. External evaluator
  feedback is evidence, not automatically a complete definition of success: a narrow
  benchmark can fully operationalize its matching goal, while a broad natural-language
  goal may require the harness to develop additional tests or evaluators. The harness
  judges that sufficiency and is explicitly told not to create evaluators ritualistically;
- runs with no stated goal use an **open-ended base prompt**: observe maps evidence and
  uncertainty, propose chooses a self-directed experiment/question/change, and reflect
  records effects and surprises without calling them improvements. The harness may later
  formulate its own provisional goals and evaluators as evolved state;
- the **goal text** (freeform, decoupled from evaluators) joins **all four phases** because
  a fresh observe or propose otherwise investigates and plans against the wrong objective;
- **last episode's OBSERVE-visible evaluator feedback** joins the **observe** phase
  (HIDDEN results never appear here);
- the **disposition's phase text** joins each phase — *unless* the adapter declares
  `disposition_in_files = True` (dsh/pi carry the perturbation in `AGENTS.md`; adding
  the prompt copy would double the dose, through a channel outside `F`).
- for `continuity_mode="framework"`, the portable handoff protocol joins every phase;
  native and deliberately independent harnesses receive no file-specific instructions.
- for `staged_activation=True`, every phase is told that `/workspace` is the frozen active
  harness and all persistent edits belong under `/workspace/candidate`; even reflect may
  inspect but must not reload candidate code.

The agent sees these texts and nothing else. It is never told why.

This framework default reaches adapters that execute `spec.phase_prompts`. Aki currently
keeps its registered no-goal prompts inside its own frozen supervisor and is the documented
legacy exception; goal injection and the default epistemic protocol are not yet wired for
that adapter.

### 2. Freeze the active harness — framework

For an adapter declaring `staged_activation=True`, Proteus materializes the last accepted
commit into a framework-private `active_root` outside the run root. `root/harness` remains
the writable candidate. This is an episode-level boundary: **observe, propose, act, and
reflect all execute the same active snapshot**. A phase never boots edits made earlier in
that episode. Keeping `active_root` outside both the candidate and the writable handoff
mount prevents the subject from modifying the frozen runtime through an alias path.

### 3. Run the episode — the adapter's core

`adapter.run_episode(spec)`, where spec carries the run root, episode number, model, the
four phase prompts, budget plan, and seed. *How* the phases execute is entirely the
adapter's:

- **minimal / llm** (in-process): the four phases run in the framework process, one
  JSONL trace line per step; the effective budget plan is enforced directly — stop
  cleanly, finish the episode;
- **dsh / pi** (external CLI): each phase boots a **fresh container**, but all four boot
  the same last-valid source. The private active snapshot mounts read-only at `/workspace`;
  the writable `root/harness` candidate mounts at `/workspace/candidate`; native state is
  at `/state`, and an optional benchmark workspace is at `/workspace/task`. The image's
  rebuild-on-boot wrapper therefore always builds the frozen active source during phases.
  DSH and Pi also bind `<run>/.proteus-state` over `/workspace/.proteus`: it remains
  writable under workspace-only permissions while staying outside the measured harness.
  Before each phase, the prior handoff is exposed; after it, an agent-written operational
  summary is archived. A budget or timeout stop falls back to normalized tool names and
  paths, never raw reasoning or tool results. History lives under
  `.proteus-state/handoffs/epNNN/`, and reflect carries into the next episode.
  The effective hard limit is enforced in two layers, both harness-agnostic: **exactly
  between phases** (no new phase once the budget is spent) and **approximately
  mid-phase** (the session log is polled live — pi's is plain JSONL, dsh's flushes one
  zstd frame per event — and the container is stopped when the count crosses the
  budget). The legacy `min_turns_per_phase` additionally reserves turns for later phases:
  while phase i runs, its stop line is
  `max_turns - min_turns_per_phase x phases_after_i`, and reaching the line ends the phase,
  not the episode — so a greedy observe cannot starve act. A budget stop records
  `turn_capped`, not an error: files already written
  persist, the episode snapshots normally, the run continues. `phase_timeout_s` remains
  the wall-clock backstop. With `announce_budget`, the agent is also *told* its budget
  in every phase prompt, so it can plan within it — off by default, because announcing
  changes behaviour, and recorded in the manifest. The phase-aware protocol below is the
  recommended configuration for longer source-evolution work.
- **aki**: delegates the episode to Aki's own supervisor.

#### Phase-aware budget protocol v1

`max_turns` remains the **normal planned budget**, so existing configurations retain their
meaning. An optional explicit plan adds three knobs:

```bash
--max-turns 300 \
--phase-turns observe=40,propose=25,act=200,reflect=35 \
--hard-max-turns 500 \
--checkpoint-turns 2 \
--announce-budget
```

- The four `--phase-turns` values must name every phase, sum exactly to `--max-turns`,
  and cannot be combined with `--min-turns-per-phase`.
- Observe and propose may spend only their own planned allowance. If either stops early,
  its unused calls remain in the pool instead of being consumed by the next planning
  phase.
- Act is the borrowing phase: it receives unused observe/propose calls and may use the
  difference between the normal 300-call plan and the 500-call hard ceiling. Reflect's
  planned 35 calls stay protected. With full early phases the cumulative stop lines are
  40, 65, 465, and 500; if an early phase stops short, act receives the difference.
- At every phase start, bundled budget-aware adapters add the current calls used, hard
  calls remaining, planned phase allowance, cumulative phase stop, and checkpoint window
  to that fresh context. The manifest records protocol version, both limits, all four
  allocations, checkpoint reserve, and `unused_priority: act`; resume locks them as part
  of the experimental condition.
- `checkpoint_turns` is an **agent-visible end-of-phase reserve**, not framework-authored
  memory. With framework continuity, the agent must use it to update
  `/workspace/.proteus/handoff.md`; Proteus archives that exact text. If it does not,
  Proteus retains the existing action-only recovery fallback and increments
  `checkpoint_misses`. It never turns traces or hidden reasoning into a semantic skill,
  memory, or plan on the harness's behalf.

The hard ceiling is enforcement; the checkpoint window is a protocol the harness must
honour. An opaque tool loop cannot have an already-executed ordinary call retroactively
converted into a handoff call, so missed checkpoints are made visible rather than silently
presented as successful memory. Custom adapters consume the same harness-neutral
`budget_plan(spec)` and `phase_prompt(spec, phase, used_before)` helpers, then enforce the
returned cumulative stop in their own native loop.

An exception or `res.ok == False` preserves the partial candidate under
`refs/proteus/candidates/episode-N-failed`, automatically restores files, index, and HEAD
to the prior valid checkpoint, and ends the trajectory so resume can retry episode N. For
a staged adapter, the preserved commit is then restored only as that retry's writable
candidate. The attempt does **not** receive an `episode N` commit and therefore does not
count as complete.

### 4. Read the trace — adapter

`adapter.read_trace(root, ep)` parses the harness's **own logs** into normalized
`ActionEvent`s (turn / phase / tool / surface / params / text). This is the only channel
Proteus reads behaviour through — never the agent's self-report, and never by
instrumenting the harness. Path→surface attribution is the adapter's mapping
(`src/…` → `loop`, etc.).

### 5. Boundary viability gate — framework + adapter

After reflect, and only then, Proteus calls the optional model-free
`validate_candidate(harness_root)`. Pi exact-syncs the candidate through its normal image
boot path, rebuilds it, and runs `--version`. DSH first recreates workspace links with a
frozen offline dependency install when its dependency-input hash changed, rebuilds and
dynamically caches every discovered package `lib/`, then runs `--version`. It next opens a
**second, fresh container** and boots the exact headless profile from that cache with an
isolated home and no provider credential.
Reaching the deterministic missing-credential boundary proves the plugin graph loaded; a
missing package, stale lockfile, uncached dependency, or headless startup error rejects the
candidate. The candidate never controls a model session during either probe.

- **pass**: the candidate may proceed to evaluators and selection, then becomes eligible
  to activate in episode N+1;
- **fail**: commit `candidate N [viability failed]`, restore the last valid state, commit
  the gapless `episode N [viability failed; rolled back]` checkpoint, record the build
  error, and continue. Episode N+1 runs healthy code, receives the failure detail, and
  restores the exact failed tree into its separate writable candidate for repair.

The gate runs before arbitrary evaluators, so invalid candidate code is not accidentally
executed by benchmark or custom evaluation either. If an adapter does not implement the
hook, Proteus has no harness-specific compile/boot command to run and this gate is skipped.

### 6. Run every evaluator — framework

`cfg.goal.evaluate(trace, ctx)` runs all evaluators **before the snapshot** (so
selection can still reject the episode):

- each evaluator carries its own kind (measurement / benchmark / custom) and visibility
  (HIDDEN / OBSERVE);
- an evaluator is user code — **its crash must not take the trajectory down**; it
  degrades to a scored zero and the run continues;
- the timing contract: between one episode's end and the next one's start, every result
  is complete.

### 7. Selection — framework

Under `selection="accept_reject"`: mean score below the best so far → reject. Selection
reads scores directly and is independent of visibility — an outer loop may act on scores
the agent itself never sees.

### 8. Snapshot / promotion — framework (the rejection semantics matter)

- **accepted**: commit `episode N` (`--allow-empty` — an episode that changed nothing
  still maps to exactly one commit; the episode→commit mapping must have no gaps). It is
  activated only when episode N+1 materializes its frozen snapshot;
- **rejected** (non-destructive):
  1. commit `candidate N [rejected]` first — the rejected tree **enters history**, the
     evidence is kept;
  2. `git restore --source` back to the last accepted state (not `reset --hard`, which
     would orphan the candidate commit);
  3. `clean -fdx` — ignored files go too, or the rejected episode's residue leaks into
     the next one;
  4. commit `episode N [rejected]`, keeping the mapping gapless.

A viability rejection uses the same preservation pattern with
`candidate N [viability failed]` and
`episode N [viability failed; rolled back]`. Unlike an infrastructure failure, it is a
completed experimental episode: the failed evolutionary proposal is part of the
trajectory. Its code is not allowed to control the next one, but staged adapters reuse it
as the next writable repair base. A valid but lower-scoring **selection** rejection does
not carry forward; the configured outer loop deliberately rejected that candidate.

Only `harness/` participates in selection. A benchmark task is the exercise rather than
the measured subject, so `<run>/task/` moves forward and is not restored when a harness
candidate is rejected.

### 9. Records and feedback — framework

- private `eval_history` appends every result plus the accept/reject flag and is written
  atomically after every snapshot checkpoint;
- numeric counters (`tokens_in` / `tokens_out`, …) sum across episodes into
  `RunResult.counters`;
- `prior_feedback` becomes the OBSERVE-visible feedback text for the next episode's
  step 1 (with a "your changes were not kept" note after a rejection);
- the progress line — which carries the condition label and **HIDDEN scores** — goes to
  `progress_path`, which **must live outside the run root**: the subject can read its
  own run root.

Evaluator history, the initial/candidate/checkpoint disposition fingerprints, and the
temporary `pending_candidate.json` repair pointer live under the sibling
`.proteus-records/<run-id>/`, never inside the subject-visible run root. Resume requires
the snapshot count, history rows, continuity checkpoint, and current fingerprint to agree
with the last durable checkpoint before it restores any pending tree on the writable side.

At sweep level, `manifest.json` carries a versioned immutable `condition` record: adapter
identity/runtime knobs, surfaces, disposition fingerprints, model, goal, evaluators, task,
budgets, continuity, and caller-supplied non-secret metadata. `--on-existing resume`
compares it before touching the run; `refuse` also checks before writing anything. A v0.1
manifest has no condition lock and is deliberately not resumable under v0.2.

### 10. Next episode — framework

Context-fresh: the next episode materializes and boots the last valid snapshot. Normally
its writable candidate starts from the same tree; after a staged viability/infrastructure
failure, only the writable side starts from the preserved failed commit. In a framework-
continuity run, the prior reflect's bounded operational handoff also crosses the boundary
as apparatus state; because it lives outside the harness snapshot, it cannot count as
evolved memory. Raw conversation and process state never survive.

---

## Shared by every harness

| part | where |
|---|---|
| versioned goal/no-goal defaults and epistemic protocol | `episode_protocol.py` |
| phase-prompt assembly rules (where goal / feedback / disposition inject) | `episode._phase_prompts` |
| phase allocation, hard ceiling, live budget prompt | `budget.py` |
| framework continuity protocol, redaction, phase history, fallback | `continuity.py` |
| evaluator timing, visibility, crash degradation, selection | `goal.py` + `episode.run` |
| snapshots, non-destructive rejection, gapless mapping, no ignore rules | `snapshot.py` |
| records: eval_history / counters / progress lines | `episode.run` |
| resume, counted in snapshot commits | `episode.completed_episodes` |
| the measurement suite — structural distance, travel, behavioural R, audit, reliability — reads the same artefacts for every harness | `proteus/measure/` |
| container infrastructure (image / network / mounts / resources, all user-configurable) | `proteus/sandbox/` |

## Owned by the adapter (seven methods, optional capability attributes)

| contract item | decides | examples of divergence |
|---|---|---|
| `name` | stable adapter identity used in records and diagnostics | minimal / dsh / pi / aki |
| `continuity_mode` | native / framework / none; absent means native | dsh/pi: framework; aki: native; minimal: none |
| `staged_activation` | request a frozen active snapshot plus separate writable candidate | True for dsh/pi |
| `surfaces()` | the measurable surface manifest (data, not a constant) | minimal: notes+tools; dsh: +instructions+loop; pi: +skills |
| `seed()` | the episode-0 state | dsh/pi extract their own code into `src/` |
| `install_disposition()` | the perturbation's carrier | minimal: a JSON file; dsh/pi: a marked `AGENTS.md` block |
| `disposition_fingerprint()` | records F drift per candidate/checkpoint; verifies resume alignment | hash of the block / file |
| `run_episode()` | how the four phases execute | in-process / one container per phase / delegated |
| `read_trace()` | each harness's log format → `ActionEvent`s | JSONL / zstd JSONL / session events |
| `required_edit_tools()` | evidence the harness can still edit itself | write / write+edit |
| `validate_candidate()` | optional model-free boundary viability gate | pi: rebuild + `--version`; dsh: frozen offline dependency check + rebuild, then fresh-container headless cold start |
| `disposition_in_files` | skip the prompt channel (double-dose guard) | True for dsh/pi |
| the self-code arrangement | how the harness's own code becomes an evolvable surface | aki: `sys.path`-first copy; dsh/pi: run-local real source rebuilt at boot |

## The built-in harnesses, side by side

| | minimal | llm | dsh | pi | aki |
|---|---|---|---|---|---|
| execution | in-process, mock policy | in-process, live model | container per phase | container per phase | delegated supervisor |
| trace source | own JSONL | own JSONL | `session.jsonl.zstd` | session JSONL | Aki tracer |
| disposition carrier | JSON file | JSON file | `AGENTS.md` block | `AGENTS.md` block | apparatus-native |
| continuity | none | none | Proteus framework handoff | Proteus framework handoff | native supervisor |
| self-code | none | none | **real TS source (staged; boundary rebuild)** | **real TS source (staged; boundary rebuild)** | `loop.py` + package copy |
| iteration bound | phase plan, direct hard stop | phase plan, direct hard stop | phase plan: exact between phases + mid-phase log watch | phase plan: exact between phases + mid-phase log watch | apparatus turn gate |
| needs | nothing | API key | Docker + key | Docker + key | the private Aki repo |

## Failure paths

| situation | outcome |
|---|---|
| a phase times out / the CLI exits nonzero | partial candidate is preserved under a dedicated ref; prior valid runtime is restored; resume retries the same episode with the partial tree as its writable repair candidate |
| the agent breaks its own code | boundary gate preserves the failed candidate, keeps the next runtime on healthy code, and restores the failed tree as the next writable repair candidate |
| one evaluator crashes or returns a non-finite score | that evaluator gets a named zero; other evaluator results survive |
| a nested `.git` appears in the harness | the episode records a snapshot error; nested metadata is refused and automatic restore removes its contents |
| the episode is rejected by selection | candidate tree preserved in history, working tree rolled back, mapping gapless |
| the process is killed mid-run | for staged adapters, resume captures the dirty tree under a failed-attempt ref, restores the last-valid runtime, and retries with that tree as writable candidate; native adapters hard-restore the checkpoint |

## Invariants to test in every staged adapter

1. Every phase of episode N executes the same active commit, even after candidate writes.
2. Active is read-only and cannot be reached through a writable handoff/task mount.
3. Reflect can inspect candidate/diff but cannot reload candidate as its own harness.
4. Validation happens once, after reflect and before arbitrary evaluators.
5. A valid candidate first controls episode N+1, never episode N.
6. A validation failure preserves evidence, restores the prior valid active tree, creates
   a gapless rejected checkpoint, and restores the failed tree only on the writable side.
7. An adapter/provider failure does not count the incomplete episode; a staged retry gets
   its preserved partial candidate while its active runtime stays at the checkpoint.
8. Resume captures staged crash-time edits before reset and restores them as the writable
   repair base. Native adapters, which cannot isolate active from candidate, discard them.
