from runs.gchill_sweep import sweep, families, sizes, baselines, variants


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