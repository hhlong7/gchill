from runs.gchill_sweep import sweep, families, baselines, variants


#a small sweep: one row per (family, policy), nothing beats the lower bound, gchill reports its solve
def test_sweep():
    df = sweep(bandwidths=[10e9], seeds=[0])
    assert len(df) == len(families) * (len(baselines) + len(variants))
    assert (df["gap"] >= -1e-9).all()
    gchill = df[df["policy"].isin(variants)]
    assert (gchill["solve_ms"] >= 0).all()
    assert gchill[gchill["policy"].str.startswith("g_ilp")]["ilp_T_ns"].gt(0).all()