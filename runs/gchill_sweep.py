"""
gchill_sweep runs the new algorithm (flag combinations) and the baselines on the xu 2023 dag families
at 4 dag sizes across dram bandwidths, and saves one row per run to a csv

- every run gets a fresh copy of the same workload (same family + seed => same dag)
- gap = how far the makespan is above the lower bound (optimal.py), 0 = provably optimal
- solve_ms and ilp_T_ns come from gchill (empty for the baselines), n_rollouts from rollout
- the dags run in parallel on every cpu core (workers), one dag = every policy on it
run from the repo root: python -m runs.gchill_sweep
"""

import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from sim.config import MemoryConfig, SystemConfig
from sim.simulator import Simulator
from sim.metrics import summary, gap
from sim.workload import tree, fork_join, random_dag, fft, gaussian_elimination
from sim.policies.fifo import Fifo
from sim.policies.heft_greedy import HeftGreedy
from sim.policies.qilin_style import QilinStyle
from sim.policies.contention_aware import ContentionAware
from sim.policies.optimal import lower_bound
from sim.policies.gchill.ilp import ContentionILP
from sim.policies.eshcs import Eshcs


#the knobs of each family per dag size. kernels per dag: small 14-20, medium 27-40,
#large 53-121, xlarge 137-255
sizes = {
    "small": dict(levels=4, fanout=2, width=4, stages=3, nodes=20, n_levels=5, edge_prob=0.2, points=4, m=5),
    "medium": dict(levels=5, fanout=2, width=8, stages=3, nodes=40, n_levels=5, edge_prob=0.2, points=8, m=7),
    "large": dict(levels=5, fanout=3, width=12, stages=4, nodes=80, n_levels=6, edge_prob=0.2, points=16, m=10),
    "xlarge": dict(levels=8, fanout=2, width=16, stages=8, nodes=200, n_levels=10, edge_prob=0.1, points=32, m=18),
}

#rollout gets slow on big dags (about n^2), so fewer seeds there
seeds = {"small": range(10), "medium": range(10), "large": range(10), "xlarge": range(3)}


#family name -> make(seed) at one size
def families(size):
    z = sizes[size]
    return {
        "tree": lambda seed: tree(levels=z["levels"], fanout=z["fanout"], seed=seed),
        "in_tree": lambda seed: tree(levels=z["levels"], fanout=z["fanout"], inward=True, seed=seed),
        "fork_join": lambda seed: fork_join(width=z["width"], stages=z["stages"], seed=seed),
        "random": lambda seed: random_dag(n_nodes=z["nodes"], n_levels=z["n_levels"], edge_prob=z["edge_prob"], seed=seed),
        "fft": lambda seed: fft(points=z["points"], seed=seed),
        "gauss": lambda seed: gaussian_elimination(m=z["m"], seed=seed),
    }


baselines = {"fifo": Fifo, "heft": HeftGreedy, "qilin": QilinStyle, "contention_aware": ContentionAware, "eshcs": Eshcs}

#gchill flag combinations, one step at a time (the e4 ablation grows here)
variants = {
    "g_base": dict(ilp=False, adjusted=False), #fastest alone plan, standalone costs
    "g_adjusted": dict(ilp=False, adjusted=True), #+ costs with the expected dram sharing
    "g_ilp": dict(ilp=True, adjusted=False), #+ ilp plan on standalone costs
    "g_ilp_adjusted": dict(ilp=True, adjusted=True), #ilp + adjusted (the current default)
    "g_ilp_adjusted_eshcs": dict(ilp=True, adjusted=True, priority="eshcs"),
    "g_ilp_adjusted_lb": dict(ilp=True, adjusted=True, priority="lower_bound"),
    "g_static": dict(ilp=False, adjusted=False, override=False),
    "g_ilp_static": dict(ilp=True, adjusted=False, override=False),
    #rollout on top of g_base, the best greedy so far
    "g_roll1": dict(ilp=False, adjusted=False, rollout_k=1),
    "g_roll2": dict(ilp=False, adjusted=False, rollout_k=2),
    "g_roll3": dict(ilp=False, adjusted=False, rollout_k=3),
    "g_roll_full": dict(ilp=False, adjusted=False, rollout_k="full"),
}

skip = {("xlarge", "g_roll_full")} #(size, policy) pairs too slow to run

bandwidths = [5e9, 10e9, 25e9, 50e9, 100e9] #low -> high contention is right -> left


def make_policy(name, kernels):
    if name in baselines:
        return baselines[name]()
    return ContentionILP(kernels, **variants[name])


#one policy on one workload, returns one row
def run_one(name, make, seed, config, lb):
    ks = make(seed) #a fresh copy, a run fills in start/finish on the kernels
    policy = make_policy(name, ks)
    sim = Simulator(config, policy)
    sim.submit(ks)
    sim.run()
    s = summary(sim)
    return {
        "policy": name,
        "makespan_ns": s["makespan_ns"],
        "lower_bound_ns": lb,
        "gap": gap(s["makespan_ns"], lb),
        "dram_util": s["dram_util"],
        "accel_util": s["accel_util"],
        "sched_per_pick_ns": s["sched_per_pick_ns"],
        "solve_ms": getattr(policy, "solve_ms", None),
        "ilp_T_ns": getattr(policy, "T", None),
        "n_rollouts": getattr(policy, "n_rollouts", None),
    }


#every policy on one dag. a top level function so the worker processes can run it
def run_dag(size, bw, fam, seed, policies):
    make = families(size)[fam]
    config = SystemConfig(memory=MemoryConfig(dram_bw_bytes_per_s=bw))
    lb = lower_bound(make(seed), config)
    n = len(make(seed))
    return [{"size": size, "bw_gbps": bw / 1e9, "family": fam, "seed": seed, "n_kernels": n,
             **run_one(name, make, seed, config, lb)}
            for name in policies if (size, name) not in skip]


#every (size, bandwidth, family, seed, policy). only the dram bandwidth changes in the default system.
#workers = how many processes, 1 = all in this one
def sweep(sizes=tuple(sizes), bandwidths=bandwidths, seeds=seeds, policies=None, workers=os.cpu_count()):
    policies = policies or [*baselines, *variants]
    jobs = [(size, bw, fam, seed, policies) for size in sizes for bw in bandwidths
            for fam in families(size) for seed in seeds[size]]
    if workers == 1:
        rows = [run_dag(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(run_dag, *zip(*jobs)))
    return pd.DataFrame([row for dag in rows for row in dag])


#mean gap in %, one row per policy
def gap_table(df, by):
    return (df.pivot_table(index="policy", columns=by, values="gap", aggfunc="mean") * 100).round(1)


if __name__ == "__main__":
    df = sweep()
    os.makedirs("data/results", exist_ok=True)
    df.to_csv("data/results/gchill_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(df)} runs -> data/results/gchill_sweep.csv\n")
    for by in ["size", "bw_gbps", "family"]:
        print(f"mean gap above the lower bound (%), by {by}")
        print(gap_table(df, by), "\n")