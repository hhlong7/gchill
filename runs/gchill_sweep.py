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
import sys
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
from sim.policies.gchill.ilp import ContentionILP, best, best_rollout
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
    #(4) heuristics: malleable cores, on its own and under rollout
    "g_malleable": dict(malleable=True),
    "g_roll1_malleable": dict(rollout_k=1, malleable=True),
    "g_roll3_malleable": dict(rollout_k=3, malleable=True),
        #(4) heuristics on the greedy, one at a time (feeding, harm and pairing lost: kept for e4)
    "g_harm": dict(harm=True),
    "g_pairing": dict(pairing=True),
    "g_feed": dict(feed=True),
    "g_delay": dict(delay=True),
    "g_delay_trigger": dict(delay=True, trigger=True),
    #the final configs (ilp.py presets)
    "gchill": best,
    "gchill_rollout": best_rollout,
}

#e4 ablation, starting from eshcs: {row: (the row it adds to, gchill flags, what kind of step)}.
#the main chain keeps each piece; a branch is a piece that is tried on its parent and not kept.
#kind: "contention" = only matters because dram is shared, "general" = would help without contention,
#"search" = rollout (sees contention through the real rate model, software only)
e4 = {
    "e4_static": ("eshcs", dict(override=False, priority="eshcs"), "general"),
    "e4_override": ("e4_static", dict(priority="eshcs"), "general"),
    "e4_upward": ("e4_override", dict(), "general"),
    "e4_malleable": ("e4_upward", dict(malleable=True), "general"),
    "e4_delay": ("e4_malleable", dict(malleable=True, delay=True), "contention"),
    "e4_trigger": ("e4_delay", dict(malleable=True, delay=True, trigger=True), "contention"),
    "e4_roll1": ("e4_trigger", dict(malleable=True, delay=True, trigger=True, rollout_k=1), "search"),
    "e4_roll3": ("e4_roll1", dict(malleable=True, delay=True, trigger=True, rollout_k=3), "search"),
    #branches
    "e4_lb_rank": ("e4_override", dict(priority="lower_bound"), "contention"),
    "e4_ilp": ("e4_upward", dict(ilp=True), "contention"),
    "e4_adjusted": ("e4_upward", dict(adjusted=True), "contention"),
    "e4_harm": ("e4_malleable", dict(malleable=True, harm=True), "contention"),
    "e4_pairing": ("e4_malleable", dict(malleable=True, pairing=True), "contention"),
    "e4_feed": ("e4_malleable", dict(malleable=True, feed=True), "general"),
}


#one row per e4 step: its paired change against the row it adds to (compare), mean over all the dags
#and per bandwidth, and its decision cost
def e4_table(df):
    rows = {}
    us = df.groupby("policy")["sched_per_pick_ns"].mean() / 1e3
    for name, (parent, _, kind) in e4.items():
        c = compare(df, name, parent)
        rows[name] = {"vs": parent, "kind": kind, **c.loc["all"].to_dict(),
                      **{i: c.loc[i, "mean"] for i in c.index if i.startswith("bw")}, "us_per_pick": us[name]}
    return pd.DataFrame(rows).T

skip = {("xlarge", "g_roll_full"), ("xlarge", "e4_roll3"), ("xlarge", "gchill_rollout")} #(size, policy) pairs too slow to run

bandwidths = [5e9, 10e9, 25e9, 50e9, 100e9] #low -> high contention is right -> left


def make_policy(name, kernels):
    if name in baselines:
        return baselines[name]()
    if name in e4:
        return ContentionILP(kernels, **e4[name][1])
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
    cores = tuple(range(1, config.cores.n_cores + 1)) #every core count a policy may give a kernel
    lb = lower_bound(make(seed), config, core_options=cores)
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

#is a better than b? paired on the same dags: change = a / b - 1 per dag, in %. one row for all the
#dags, then one per bandwidth and one per size. win/loss = % of dags where a is faster/slower
def compare(df, a, b):
    ms = df.pivot_table(index=["size", "bw_gbps", "family", "seed"], columns="policy", values="makespan_ns")
    change = (ms[a] / ms[b] - 1) * 100
    def row(c):
        return {"mean": c.mean(), "win": (c < -1e-9).mean() * 100, "loss": (c > 1e-9).mean() * 100, "worst": c.max()}
    rows = {"all": row(change)}
    for level, name in [("bw_gbps", "bw"), ("size", "size")]:
        for key, c in change.groupby(level=level):
            rows[f"{name} {key}"] = row(c)
    return pd.DataFrame(rows).T.round(2)


#the dags each phase check runs on: 3 sizes x 5 seeds (xlarge is too slow for a quick check)
quick = dict(sizes=("small", "medium", "large"), seeds={size: range(5) for size in ("small", "medium", "large")})

#mean gap in %, one row per policy
def gap_table(df, by):
    return (df.pivot_table(index="policy", columns=by, values="gap", aggfunc="mean") * 100).round(1)


#python -m runs.gchill_sweep compare a b => the quick dags, only a and b, then compare(a, b)
if __name__ == "__main__" and sys.argv[1:2] == ["compare"]:
    a, b = sys.argv[2], sys.argv[3]
    print(compare(sweep(policies=[a, b], **quick), a, b))

#python -m runs.gchill_sweep e4 => eshcs + every e4 row on every dag size, then the step table
elif __name__ == "__main__" and sys.argv[1:2] == ["e4"]:
    df = sweep(policies=["eshcs", *e4])
    os.makedirs("data/results", exist_ok=True)
    df.to_csv("data/results/e4.csv", index=False)
    pd.set_option("display.width", 200)
    print(e4_table(df).round(2))

elif __name__ == "__main__":
    df = sweep()
    os.makedirs("data/results", exist_ok=True)
    df.to_csv("data/results/gchill_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(df)} runs -> data/results/gchill_sweep.csv\n")
    for by in ["size", "bw_gbps", "family"]:
        print(f"mean gap above the lower bound (%), by {by}")
        print(gap_table(df, by), "\n")
