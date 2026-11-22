# Open-Source Integration Plan

**Goal:** make every result in the project as close to real hardware as we can afford, by running the
schedulers on open-source, widely used simulators and on open-source workloads, and by checking each
modeling assumption against a more detailed tool.

Status (2026-11-22): planned, not started.

## Why

Two weaknesses in the current results:

1. **The workloads are too small.** A whole DAG from the Xu 2023 families moves 0.4-4.8 MB, less than the
   12 MB LLC in our own config. On real hardware most of that traffic would hit in the cache, so a
   reviewer can question whether there is any DRAM contention to schedule around.
2. **Every number comes from our own simulator.** Its DRAM model (proportional sharing, no loss when
   many agents share DRAM) has never been checked against a DRAM simulator, and the made-up loss rates
   we tried change the conclusions (see "Sensitivity" below).

## What gets replaced, and with what

| Part | Today | After integration | What it buys |
|---|---|---|---|
| Workloads | 5 synthetic Xu 2023 families, tiny kernels | Tiled Cholesky / LU / QR, DNN operator graphs (torch.fx), STG (Kasahara) | Realistic sizes (data far larger than the LLC), standard benchmarks |
| Event engine + bandwidth sharing | `sim/simulator.py`, `sim/memory.py` | SimGrid (Python bindings), policies run live inside it | An independent, widely validated engine |
| DRAM behavior | Proportional share, no loss | Ramulator 2, queried for each mix of running kernels (cached), plus whole-DAG request-level replays | Measured loss under load and measured sharing between streams |
| Cores, caches, accelerator timing | `sim/cores.py`, `sim/systolic.py` (analytical) | Unchanged for the sweeps; checked on gem5 with a systolic accelerator and Ramulator 2 as memory | A cycle-level sanity check of the policy ranking |

What stays ours: the policies, the kernel timing model (`cores.py`, `systolic.py`), and our simulator as the
**policies' own predictor** (rollout and the delay check need to simulate ahead, and nobody can copy a
running SimGrid or gem5). The run uses the realistic models; the policy's predictions do not. That is the
realistic setup: a scheduler's model is never the hardware.

## Tools

| Tool | Role | Why this one | Fallback |
|---|---|---|---|
| **SimGrid** | Main engine for every sweep | Python actors (policies run unchanged), multicore hosts, max-min bandwidth sharing, a NONLINEAR link-sharing callback in Python (for DRAM loss under load), used in many published studies | none needed at this level; StarPU-SimGrid, DS3 and SST were considered and rejected (C-only schedulers, no shared-DRAM contention, no task runtime) |
| **Ramulator 2** (CMU SAFARI, MIT license) | DRAM model | DDR5 / LPDDR5 / HBM3 / GDDR6, standalone trace mode, also usable as gem5's memory, modular (plugins for frontends and request schedulers) | DRAMsim3 (no DDR5/LPDDR5, needs a forked gem5), DRAMSys (heavier) |
| **gem5** + systolic accelerator | Cycle-level check on a few DAGs | Real cores, caches, DMA; Ramulator 2 as its memory | Gem5-AcceSys (MatrixFlow systolic array) if its code is public, else gem5-SALAM; Chipyard + Gemmini rejected (RTL, far too slow) |
| **Workloads** | Benchmarks | Tiled factorizations are the standard CPU+accelerator scheduling benchmark (StarPU, PaRSEC); DNN graphs give real GEMM shapes; STG gives comparability with the scheduling literature | WfCommons (real workflows, but CPU tasks only) kept as an option |

## How the pieces connect

```
 generators / importers                  one JSON file per DAG
 (tiled LA, torch.fx, STG)  ----------->  (kernels, sizes, deps, stated rules)
                                                   |
                                                   v
 policies (gchill, HEFT, ESHCS, ...)  <-->  system view  <-->  backend
   rollout / delay use our model                               |-- our simulator (fast, predictor)
   as an internal predictor                                    |-- SimGrid  --> DRAM service --> Ramulator 2
                                                               |                (cached by running mix)
                                                               '-- schedule export --> gem5 replay
                                                                                  (C dispatcher, Ramulator 2 memory)
```

- **Workload file:** the one format every tool reads. Anything a source does not provide (STG has no
  data sizes; DNN ops need a CPU/accelerator mapping; data addresses for DRAM) is assigned by a stated,
  seeded rule written into the file.
- **System view:** the small interface the policies read (time, ready and running kernels, free cores,
  accelerator state, each running kernel's progress, config). Today policies read simulator internals
  directly; after the refactor any backend can serve them.
- **SimGrid backend:** builds the machine (CPU host with n cores, accelerator host, DRAM link), turns each
  kernel's segments into SimGrid activities, and runs a scheduler actor that calls `pick()` at every
  arrival and finish.
- **DRAM service:** at every change in the set of running kernels, looks up the bandwidth each one gets
  for that exact mix (access pattern, read/write ratio, demand, priority order). On a cache miss it runs
  Ramulator 2 on a short window of that mix. The cache persists across all runs.
- **gem5 replay:** a policy cannot run live inside gem5, so a run on SimGrid exports its decisions
  (resource, cores, start order per resource) and a small C dispatcher on the simulated cores replays them.

## Fidelity: what is real, what is still assumed

| Aspect | How it is modeled after integration | Remaining gap | How the gap is measured |
|---|---|---|---|
| DRAM bandwidth under load | Ramulator 2 per running mix | Fluid rate between events (no per-request latency or bursts) | Whole-DAG request-level replay in Ramulator, per-kernel error |
| Sharing between streams | Ramulator 2 per-stream results (row-hit favoritism included) | Binning of the mix key | Halve the bins, makespan change must stay < 1% |
| Data placement | Stated layout rule (each buffer contiguous, separate regions) | Real allocators differ | Sensitivity run with a second layout rule |
| Caches | Not in SimGrid; traffic treated as post-LLC | Reuse between producer and consumer | gem5 replay (has caches) |
| CPU and accelerator speed | Our analytical models | Microarchitecture effects | gem5 replay |
| Scheduler decision time | Measured, not charged to makespan | Real dispatch latency | Tier 3 cost model (charge a latency per decision) |
| DRAM priority arbiter | Ramulator 2 request-scheduler plugin + SimGrid | Real controllers expose a few QoS levels only | Arbiter limited to a small number of priority levels |

## Fair baselines

Realistic workloads expose how much the baselines are handicapped. Found in a first test (below):

- `heft_greedy.py` handles ready kernels in arrival order. Textbook HEFT orders them by upward rank.
- No baseline can give one kernel several cores; gchill can (malleable cores).

Before any claim, the comparison set gets:
1. **HEFT with upward rank** (textbook), next to the current arrival-order version.
2. **HEFT with moldable tasks** (the same core-widening rule gchill uses), so "gchill wins" is not just
   "gchill may use 4 cores and the baselines may not".
3. **The DRAM arbiter for everyone** (oldest-first order for the baselines), so the paper separates the
   hardware mechanism from gchill's criticality signal.

## First evidence (our simulator, scratch, 2026-11-22)

Not results, only a direction check on our current model before the integration exists.

- **Xu 2023 families vs textbook HEFT:** gchill's 3-piece preset (override + upward + malleable) is only
  0.5% faster on average and slower on 34% of DAGs. Most of its earlier lead over "HEFT" came from the
  arrival-order baseline. With the live-criticality DRAM priority it is 3.7% faster (wins 84%).
- **Tiled Cholesky (120-364 tasks, 9-119 MB moved):** gchill is 23% faster than textbook HEFT on average,
  up to 60% at high bandwidth. The gain splits into:
  - ordering by each kernel's planned-resource time (about 10%): HEFT's mean-cost rank is misled when the
    accelerator is about 200x faster than a core on GEMMs; a known HEFT weakness, a fair win;
  - malleable cores on the CPU-only panel tasks (POTRF), which are the critical path (the rest): fair in
    principle, but it needs the moldable HEFT baseline above, and it depends on our placeholder CPU
    speed and multicore scaling.
- **DRAM loss under load (made-up rates of 5% and 10% per extra stream):** with loss, the 3-piece preset
  can lose to ESHCS (+4.6% at 10%), while the DRAM priority and the delay check gain much more
  (-8% to -15%). Which world is real is exactly what Ramulator 2 decides.

## Phases

Each phase is small, has its own tests, and the user types the code. Track A pays off on our current
simulator before any tool is installed.

**Track A: workloads**
1. Workload file format (save / load, validated at load).
2. Tiled Cholesky / LU / QR generator.
3. DNN graph export (torch.fx, run once; sweeps only read the files), multi-model mixes.
4. STG importer (stated rule for kernel types and sizes).
5. Suite pilot: pick sizes where data is at least 4x the LLC and the policies differ; freeze the suite.
5b. Fair baselines: textbook HEFT (upward rank), moldable HEFT, arbiter for every policy.

**Track B: one interface**
6. System view + snapshot (policies stop reading simulator internals; agreed with the partner, who owns
   `eshcs.py` and `optimal.py`).

**Track C: SimGrid and Ramulator 2**
7. SimGrid spike on macOS: prove multicore hosts, compute/transfer overlap, the NONLINEAR callback,
   per-activity priority or rate bounds, and the Python scheduler loop.
8. SimGrid backend.
9. Cross-validation: our simulator vs SimGrid on the same runs; explain every gap (expect one from
   proportional vs max-min sharing).
10. Ramulator 2 setup: rate-limited frontend plugin, fitted table for the policies' predictors.
11. Ramulator in the loop, cached by running mix; whole-DAG replays as the closest check.
12. DRAM priority arbiter in SimGrid and as a Ramulator 2 scheduler plugin.
13. E3 / E4 rerun on the frozen suite.

**Track D: gem5 check (last, can be dropped)**
14. gem5 in Docker (arm64 Linux): multicore CPU, caches, systolic accelerator, Ramulator 2 as memory.
15. Schedule replay on 10-20 scaled-down DAGs; compare the policy ranking with SimGrid.

**Stopping points** (each one is a complete set of results):
- After 1-6 and 10: realistic workloads, fair baselines, a measured DRAM model.
- After 7-13: independent engine, DRAM close to Ramulator, the arbiter, full experiments.
- After 14-15: the cycle-level check.

## Effort

| Track | Hours (user typing, incl. debugging) |
|---|---|
| A: workloads + fair baselines | 30-50 |
| B: system view | 10-15 |
| C: SimGrid | 40-70 |
| C: Ramulator 2 (setup, in the loop, arbiter plugin) | 55-95 |
| Experiments | 10-15 + compute |
| D: gem5 | 80-150 |
| **Total** | **about 225-395** (145-245 without gem5) |

Code: about 300-500 of the current ~3,000 lines change (mostly Phase 6). New: about 2,000-3,000 lines of
Python, a few hundred lines of C++ (Ramulator 2 plugins) and C (gem5 dispatcher).

## Risks

1. gem5 accelerator path not settled (Gem5-AcceSys code availability, gem5-SALAM's LLVM flow), slow runs.
2. SimGrid has no macOS wheel; it builds from source (cmake, boost, pybind11) or via Homebrew.
3. Ramulator 2 plugins are C++ inside its codebase.
4. Model disagreements (Phases 9 and 11) each need a decision and probably a rerun.
5. Phase 6 changes what the partner's code reads.
6. Results may shrink once baselines are fair. That is the point of doing this: a smaller real gain is
   worth more than a large one on toy DAGs.

## References

- SimGrid: https://simgrid.org/doc/latest/Tutorial_DAG.html ;
  NONLINEAR sharing example: https://sources.debian.org/src/simgrid/4.1-1/examples/python/network-nonlinear/network-nonlinear.py
- Ramulator 2.0: https://arxiv.org/abs/2308.11030 , https://github.com/CMU-SAFARI/ramulator2
- DRAMsim3: https://github.com/umd-memsys/DRAMsim3 ; DRAMSys: https://github.com/Arteris-IP/DRAMSys
- Validating gem5's memory models: https://arch.cs.ucdavis.edu/assets/papers/validating_gem5_memory.pdf
- Gem5-AcceSys: https://arxiv.org/abs/2502.12273 ; Gemmini: https://arxiv.org/abs/1911.09925
- STG (Tobita and Kasahara): https://www.kasahara.cs.waseda.ac.jp/schedule/
- StarPU SimGrid mode: https://starpu.gitlabpages.inria.fr/
- Tiled factorizations on hybrid CPU-GPU systems (ICL): https://hpcg.icl.utk.edu/files/publications/2015/icl-utk-775-2015.pdf
