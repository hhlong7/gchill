import pytest
from sim.config import MemoryConfig, SystemConfig
from sim.simulator import Simulator
from sim.workload import tree, fork_join, random_dag, fft, gaussian_elimination
from sim.policies.heft import Heft, earliest_gap, topo
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator


def run(ks, config):
    p = Heft()
    sim = Simulator(config, p)
    sim.submit(ks)
    sim.run()
    return p, sim


#with no contention the standalone costs are exact, so every kernel finishes when heft planned it
@pytest.mark.parametrize("make", [lambda: tree(seed=1), lambda: fork_join(seed=2), lambda: random_dag(seed=3),
                                  lambda: fft(seed=4), lambda: gaussian_elimination(seed=5)])
def test_runs_exactly_as_planned_without_contention(make):
    ks = make()
    p, sim = run(ks, SystemConfig(memory=MemoryConfig(dram_bw_bytes_per_s=1e18)))
    for k in ks:
        assert k.finish == pytest.approx(p.planned_finish[k.kernel_id], rel=1e-9, abs=1e-6)


#rank = mean cost over the processors that can run it + the largest child rank
def test_upward_rank():
    ks = [make_gmm_kernel(0, 64, 64, 64, 0.0), make_simple_kernel(1, 1000, 0.0, deps=(0,))]
    p, sim = run(ks, SystemConfig())
    n = sim.config.cores.n_cores
    c0 = {kind: p.cost(ks[0], kind, sim) for kind in (cpu, accelerator)}
    r1 = p.cost(ks[1], cpu, sim)
    assert p.rank[1] == pytest.approx(r1)
    assert p.rank[0] == pytest.approx((n * c0[cpu] + c0[accelerator]) / (n + 1) + r1)


#insertion: a short kernel fills an idle gap, a long one goes after the last slot
def test_earliest_gap():
    slots = [(0, 10), (30, 40)]
    assert earliest_gap(slots, 0, 5) == 10
    assert earliest_gap(slots, 0, 25) == 40
    assert earliest_gap(slots, 12, 18) == 12
    assert earliest_gap([], 7, 3) == 7


def test_topo_puts_deps_first():
    ks = random_dag(seed=9)
    pos = {kid: i for i, kid in enumerate(topo(ks))}
    assert all(pos[d] < pos[k.kernel_id] for k in ks for d in k.deps)


#under contention the plan's times are wrong but every kernel still runs, deps first
def test_finishes_under_contention():
    ks = random_dag(seed=0)
    p, sim = run(ks, SystemConfig(memory=MemoryConfig(dram_bw_bytes_per_s=5e9)))
    by = {k.kernel_id: k for k in ks}
    assert all(k.done for k in ks)
    assert all(by[d].finish <= k.start + 1e-9 for k in ks for d in k.deps)
