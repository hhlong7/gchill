from runs.gchill_sweep import sweep, families, sizes, baselines, variants, compare, e4, e4_table

#a small sweep in this process: one row per (family, policy), nothing beats the lower bound,
#gchill reports its solve, rollout reports how many options it played out
def test_sweep():
    df = sweep(sizes=["small"], bandwidths=[10e9], seeds={"small": [0]}, workers=1)
    assert len(df) == len(families("small")) * (len(baselines) + len(variants))
    assert (df["gap"] >= -1e-9).all()
    gchill = df[df["policy"].isin(variants)]
    assert (gchill["solve_ms"] >= 0).all()
    assert gchill[gchill["policy"].str.startswith("g_ilp")]["ilp_T_ns"].gt(0).all()
    assert gchill[gchill["policy"].str.startswith("g_roll")]["n_rollouts"].gt(0).all()


#every family gets bigger with each size
def test_sizes_grow():
    for fam in families("small"):
        n = [len(families(size)[fam](0)) for size in sizes]
        assert n == sorted(n) and len(set(n)) == len(n), fam

#a policy against itself: no change on any dag. heft against fifo: faster on average
def test_compare():
    df = sweep(sizes=["small"], bandwidths=[10e9, 100e9], seeds={"small": [0]}, policies=["fifo", "heft"], workers=1)
    same = compare(df, "heft", "heft")
    assert (same[["mean", "win", "loss", "worst"]] == 0).all().all()
    assert list(same.index) == ["all", "bw 10.0", "bw 100.0", "size small"]
    vs = compare(df, "heft", "fifo")
    assert vs.loc["all", "mean"] < 0 and vs.loc["all", "win"] + vs.loc["all", "loss"] <= 100


#every e4 row runs and its parent is a row too (or eshcs, where it starts), so the step table is complete
def test_e4():
    assert all(parent in e4 or parent == "eshcs" for parent, _, _ in e4.values())
    df = sweep(sizes=["small"], bandwidths=[25e9], seeds={"small": [0]}, policies=["eshcs", *e4], workers=1)
    table = e4_table(df)
    assert list(table.index) == list(e4)
    assert (df["gap"] >= -1e-9).all()