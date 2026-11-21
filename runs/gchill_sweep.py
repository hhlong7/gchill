"""
gchill_sweep runs the new algorithm (flag combinations) and the baselines on the 5 dag families
across dram bandwidths, and saves one row per run to a csv

- every run gets a fresh copy of the same workload (same family + seed => same dag)
- gap = how far the makespan is above the lower bound (optimal.py), 0 = provably optimal
- solve_ms and ilp_T_ns come from gchill (empty for the baselines)
run from the repo root: python -m runs.gchill_sweep
"""

import os
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


#family name -> make(seed), the default sizes from workload.py
families = {
    "tree": lambda seed: tree(seed=seed),
    "in_tree": lambda seed: tree(inward=True, seed=seed),
    "fork_join": lambda seed: fork_join(seed=seed),
    "random": lambda seed: random_dag(seed=seed),
    "fft": lambda seed: fft(seed=seed),
    "gauss": lambda seed: gaussian_elimination(seed=seed),
}

baselines = {"fifo": Fifo, "heft": HeftGreedy, "qilin": QilinStyle, "contention_aware": ContentionAware}

#gchill flag combinations, one step at a time (the e4 ablation grows here)
variants = {
    "g_base": dict(ilp=False, adjusted=False), #fastest alone plan, standalone costs
    "g_adjusted": dict(ilp=False, adjusted=True), #+ costs with the expected dram sharing
    "g_ilp": dict(ilp=True, adjusted=False), #+ ilp plan on standalone costs
    "g_ilp_adjusted": dict(ilp=True, adjusted=True), #ilp + adjusted (the current default)
    "g_ilp_adjusted_eshcs": dict(ilp=True, adjusted=True, priority="eshcs"),
    "g_ilp_adjusted_lb": dict(ilp=True, adjusted=True, priority="lower_bound"),
}

bandwidths = [5e9, 10e9, 25e9, 50e9, 100e9] #low -> high contention is right -> left
seeds = range(10)


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
    }


#every (bandwidth, family, seed, policy), only the dram bandwidth changes in the default system
def sweep(bandwidths=bandwidths, families=families, seeds=seeds, policies=None):
    policies = policies or [*baselines, *variants]
    rows = []
    for bw in bandwidths:
        config = SystemConfig(memory=MemoryConfig(dram_bw_bytes_per_s=bw))
        for fam, make in families.items():
            for seed in seeds:
                lb = lower_bound(make(seed), config)
                for name in policies:
                    row = run_one(name, make, seed, config, lb)
                    rows.append({"bw_gbps": bw / 1e9, "family": fam, "seed": seed, **row})
    return pd.DataFrame(rows)


#mean gap in %, one row per policy
def gap_table(df, by):
    return (df.pivot_table(index="policy", columns=by, values="gap", aggfunc="mean") * 100).round(1)


if __name__ == "__main__":
    df = sweep()
    os.makedirs("data/results", exist_ok=True)
    df.to_csv("data/results/gchill_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(df)} runs -> data/results/gchill_sweep.csv\n")
    print("mean gap above the lower bound (%), by dram bandwidth (GB/s)")
    print(gap_table(df, "bw_gbps"), "\n")
    print("mean gap above the lower bound (%), by family")
    print(gap_table(df, "family"))