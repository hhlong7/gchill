import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator, fresh, topological
from sim.simulator import Simulator
from sim.workload import tree, fork_join, random_dag, fft, gaussian_elimination
from sim.policies.base import Policy
from sim.policies.gchill.ilp import ContentionILP, best, best_rollout
from sim.policies.gchill.heuristics import feed_bonus, pairing_penalty, should_delay, contended
from sim.policies.gchill.rollout import play_out
from sim.policies.gchill.costs import cost_table, adjusted_cost_table, shared_ns
from sim.policies.gchill.plan import core_plan, ilp_plan
from sim.policies.gchill.priority import descendants, upward_rank, lower_bound_rank, eshcs_rank, PRIORITY
from sim.policies.optimal import lower_bound
from sim.policies.cost_model import segments
from sim.metrics import makespan


#same system as the other tests: 2 cores at 10 gflops, 32 x 32 array at 1 GHz, 10 GB/s dram
def make_system(n_cores=2):
    return SystemConfig(
        cores=CoreConfig(n_cores=n_cores, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=10e9),
    )


def run(kernels, policy, config=None):
    sim = Simulator(config or make_system(), policy)
    sim.submit(kernels)
    sim.run()
    return sim


families = [
    lambda: tree(seed=0),
    lambda: tree(inward=True, seed=1),
    lambda: fork_join(seed=0),
    lambda: random_dag(seed=0),
    lambda: fft(seed=0),
    lambda: gaussian_elimination(seed=0),
]


#a gemm that is faster on the accelerator goes there, an elementwise kernel can only go on the cpu
def test_core_plan():
    hw = Simulator(make_system(), Policy())
    ks = [make_gmm_kernel(0, 10, 32, 32, arrival=0.0), make_simple_kernel(1, 1000, arrival=0.0)]
    assert core_plan(ks, cost_table(ks, hw)) == {0: accelerator, 1: cpu}


#chain 0 -> 1 -> 2 of 400 ns kernels: the rank is the time left down the chain
def test_upward_rank_chain():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0),
          make_simple_kernel(1, 1000, arrival=0.0, deps=(0,)),
          make_simple_kernel(2, 1000, arrival=0.0, deps=(1,))]
    cost = cost_table(ks, hw)
    rank = upward_rank(ks, core_plan(ks, cost), cost)
    assert rank == pytest.approx({0: 1200.0, 1: 800.0, 2: 400.0})


#1 core. kernel 0 is alone (rank 1200), kernel 1 has a long child (rank 3600) => kernel 1 goes first
#even though its listed 2nd, then its child (rank 3200) still beats kernel 0. order: 1, 2, 0
def test_critical_path_first():
    ks = [make_simple_kernel(0, 3000, arrival=0.0),                 #1200 ns, nothing after it
          make_simple_kernel(1, 1000, arrival=0.0),                 #400 ns, then
          make_simple_kernel(2, 8000, arrival=0.0, deps=(1,))]      #3200 ns
    run(ks, ContentionILP(ks), make_system(n_cores=1))
    assert ks[1].start == 0.0
    assert ks[2].start == pytest.approx(ks[1].finish)
    assert ks[0].start == pytest.approx(ks[2].finish)


#small gemm planned on the accelerator, but a big gemm holds it for ~2.6 us:
#the free cpu (410 ns) beats waiting => override puts it on the cpu, without override it waits
@pytest.mark.parametrize("override, where", [(True, cpu), (False, accelerator)])
def test_override(override, where):
    ks = [make_gmm_kernel(0, 64, 64, 64, arrival=0.0), make_gmm_kernel(1, 2, 32, 32, arrival=10.0)]
    run(ks, ContentionILP(ks, override=override, ilp=False))
    assert ks[0].resource == accelerator
    assert ks[1].resource == where
    if where == accelerator:
        assert ks[1].start == pytest.approx(ks[0].finish)


#without override every kernel runs where the plan put it
@pytest.mark.parametrize("make", families)
def test_follows_the_plan(make):
    ks = make()
    policy = ContentionILP(ks, override=False)
    sim = run(ks, policy)
    assert all(sim.runs[k.kernel_id].kind == policy.plan[k.kernel_id] for k in ks)


#every kernel finishes, after all its deps, and no schedule beats the lower bound, for every priority
@pytest.mark.parametrize("priority", list(PRIORITY))
@pytest.mark.parametrize("make", families)
def test_valid_schedule(make, priority):
    ks = make()
    sim = run(ks, ContentionILP(ks, priority=priority))
    by_id = {k.kernel_id: k for k in ks}
    assert all(k.done for k in ks)
    for k in ks:
        assert all(k.start >= by_id[d].finish - 1e-9 for d in k.deps)
    assert sim.now >= lower_bound(fresh(ks), make_system()) * (1 - 1e-9)


#a gemm can use both resources, an elementwise kernel only the cpu
def test_cost_table():
    hw = Simulator(make_system(), Policy())
    ks = [make_gmm_kernel(0, 10, 32, 32, arrival=0.0), make_simple_kernel(1, 1000, arrival=0.0)]
    cost = cost_table(ks, hw)
    assert cost[0] == pytest.approx({cpu: 2048.0, accelerator: 332.8})
    assert cost[1] == pytest.approx({cpu: 400.0})


#diamond 0 -> (1, 2) -> 3: kernel 3 is reached twice from 0 but counted once
def test_descendants():
    ks = [make_simple_kernel(0, 1000, arrival=0.0),
          make_simple_kernel(1, 1000, arrival=0.0, deps=(0,)),
          make_simple_kernel(2, 1000, arrival=0.0, deps=(0,)),
          make_simple_kernel(3, 1000, arrival=0.0, deps=(1, 2))]
    assert descendants(ks) == {0: {1, 2, 3}, 1: {3}, 2: {3}, 3: set()}


#kernel 0 (400 ns) fans out to 6 memory bound kernels, kernel 7 starts a chain of 3 (1200 ns).
#upward rank only sees chain length => the chain goes first. lower bound sees that 0 and its
#children move 7 x 4000 bytes = 2800 ns at 10 GB/s => kernel 0 goes first
def test_lower_bound_rank_sees_traffic():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0)]
    ks += [make_simple_kernel(i, 1000, arrival=0.0, deps=(0,)) for i in range(1, 7)]
    ks += [make_simple_kernel(7, 1000, arrival=0.0),
           make_simple_kernel(8, 1000, arrival=0.0, deps=(7,)),
           make_simple_kernel(9, 1000, arrival=0.0, deps=(8,))]
    cost = cost_table(ks, hw)
    plan = core_plan(ks, cost)
    up = upward_rank(ks, plan, cost)
    lb = lower_bound_rank(ks, plan, cost, hw)
    assert up[0] == pytest.approx(800.0) and up[7] == pytest.approx(1200.0)
    assert lb[0] == pytest.approx(2800.0) and lb[7] == pytest.approx(1200.0)


#the lower bound rank is never below the upward rank (its 1st term). the ranks need topological order
@pytest.mark.parametrize("make", families)
def test_lower_bound_at_least_upward(make):
    hw = Simulator(make_system(), Policy())
    ks = topological(make())
    cost = cost_table(ks, hw)
    plan = core_plan(ks, cost)
    up = upward_rank(ks, plan, cost)
    lb = lower_bound_rank(ks, plan, cost, hw)
    assert all(lb[kid] >= up[kid] - 1e-9 for kid in up)


#chain 0 -> 1 -> 2 -> 3 next to a fan out 4 -> (5, 6, 7), all 400 ns. on the critical path urgency is 1,
#a chain kernel has 1 child out of max 3 so each level adds only a 1/3 of the one below it
def test_eshcs_rank():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0)]
    ks += [make_simple_kernel(i, 1000, arrival=0.0, deps=(i - 1,)) for i in range(1, 4)]
    ks += [make_simple_kernel(4, 1000, arrival=0.0)]
    ks += [make_simple_kernel(i, 1000, arrival=0.0, deps=(4,)) for i in range(5, 8)]
    cost = cost_table(ks, hw)
    rank = eshcs_rank(ks, core_plan(ks, cost), cost)
    assert rank[3] == pytest.approx(1.0)
    assert rank[0] == pytest.approx(1 + 1/3 + 1/9 + 1/27)
    assert rank[4] == pytest.approx(2/3)


#1 core. kernel 0 alone (800 ns) and kernel 1 (400 ns) + its child (400 ns) have the same upward rank,
#kernel 1 has a kernel after it => it goes first even though its listed 2nd
def test_tie_break_by_descendants():
    ks = [make_simple_kernel(0, 2000, arrival=0.0),
          make_simple_kernel(1, 1000, arrival=0.0),
          make_simple_kernel(2, 1000, arrival=0.0, deps=(1,))]
    policy = ContentionILP(ks)
    run(ks, policy, make_system(n_cores=1))
    assert policy.rank[0] == pytest.approx(policy.rank[1])
    assert ks[1].start == 0.0
    assert ks[0].start == pytest.approx(ks[1].finish)


def test_unknown_priority():
    with pytest.raises(ValueError):
        ContentionILP([make_simple_kernel(0, 1000, arrival=0.0)], priority="fastest")

#a chain never runs 2 kernels at once => nothing to share, same as alone
def test_adjusted_chain_is_alone():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0),
          make_simple_kernel(1, 1000, arrival=0.0, deps=(0,)),
          make_simple_kernel(2, 1000, arrival=0.0, deps=(1,))]
    alone, adj = cost_table(ks, hw), adjusted_cost_table(ks, hw)
    for kid in alone:
        assert adj[kid] == pytest.approx(alone[kid])


#sharing dram can only slow a kernel down, and the table keeps the same kernels and resources
@pytest.mark.parametrize("make", families)
def test_adjusted_never_faster(make):
    ks = topological(make())
    hw = Simulator(make_system(), Policy())
    alone, adj = cost_table(ks, hw), adjusted_cost_table(ks, hw)
    assert adj.keys() == alone.keys()
    for kid in alone:
        assert adj[kid].keys() == alone[kid].keys()
        for where in alone[kid]:
            assert adj[kid][where] >= alone[kid][where] - 1e-6


#0 co-runners = standalone, and each extra co-runner never makes it faster
def test_more_co_runners_never_faster():
    hw = Simulator(make_system(), Policy())
    k = make_gmm_kernel(0, 64, 64, 64, arrival=0.0)
    for where in (cpu, accelerator):
        times = [shared_ns(k, where, hw, n, 5e9) for n in range(4)]
        assert times[0] == pytest.approx(cost_table([k], hw)[0][where])
        assert times == sorted(times)


#the ilp's own estimate for a plan: the biggest of its 4 terms (chain, accelerator, cpu, dram)
def estimate(kernels, plan, cost, sim):
    peak = sim.config.memory.dram_bw_bytes_per_s
    chain = max(upward_rank(kernels, plan, cost).values())
    acc = sum(cost[k.kernel_id][accelerator] for k in kernels if plan[k.kernel_id] == accelerator)
    cpu_work = sum(cost[k.kernel_id][cpu] for k in kernels if plan[k.kernel_id] == cpu)
    nbytes = sum(sum(s.bytes for s in segments(k, plan[k.kernel_id], sim)) for k in kernels)
    return max(chain, acc, cpu_work / sim.config.cores.n_cores, nbytes / peak * 1e9)


#the big gemm already fills the accelerator, so the ilp puts the small one on the free cpu,
#where fastest-alone would queue both on the accelerator
def test_ilp_balances_the_accelerator():
    hw = Simulator(make_system(), Policy())
    ks = [make_gmm_kernel(0, 64, 64, 64, arrival=0.0), make_gmm_kernel(1, 2, 32, 32, arrival=0.0)]
    cost = cost_table(ks, hw)
    assert core_plan(ks, cost) == {0: accelerator, 1: accelerator}
    plan, T = ilp_plan(ks, cost, hw)
    assert plan == {0: accelerator, 1: cpu}
    assert T == pytest.approx(estimate(ks, plan, cost, hw))


#nothing fits the accelerator => everything on the cpu
def test_ilp_cpu_only():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0), make_simple_kernel(1, 2000, arrival=0.0, deps=(0,))]
    plan, _ = ilp_plan(ks, cost_table(ks, hw), hw)
    assert plan == {0: cpu, 1: cpu}


#the ilp is optimal for its own estimate: never above fastest-alone, its T is its plan's estimate,
#and the relaxed version (x in [0, 1]) is never above the integer one
@pytest.mark.parametrize("make", families)
def test_ilp_optimal_on_its_estimate(make):
    ks = topological(make())
    hw = Simulator(make_system(), Policy())
    cost = adjusted_cost_table(ks, hw)
    plan, T = ilp_plan(ks, cost, hw)
    _, relaxed = ilp_plan(ks, cost, hw, relax=True)
    assert T == pytest.approx(estimate(ks, plan, cost, hw), rel=1e-6)
    assert T <= estimate(ks, core_plan(ks, cost), cost, hw) + 1e-6
    assert relaxed <= T + 1e-6


#ilp=False keeps the fastest-alone plan, and the policy reports the ilp's estimate and solve time
#ilp=True runs the ilp and reports its estimate and solve time, the default keeps the fastest-alone plan
def test_ilp_flag():
    ks = random_dag(seed=0)
    on = ContentionILP(ks, ilp=True)
    run(ks, on)
    assert on.T > 0 and on.solve_ms >= 0
    ks = random_dag(seed=0)
    off = ContentionILP(ks)
    run(ks, off)
    assert off.plan == core_plan(off.kernels, off.cost)
    assert off.T is None

#the big gemm holds the accelerator, 2 smaller gemms planned there arrive together. the 1st one books
#the accelerator after the big one, so the 2nd sees that queue and starts on the free cpu instead
@pytest.mark.parametrize("override, where", [(True, cpu), (False, accelerator)])
def test_override_sees_the_queue(override, where):
    ks = [make_gmm_kernel(0, 64, 64, 64, arrival=0.0),
          make_gmm_kernel(1, 16, 64, 32, arrival=10.0),
          make_gmm_kernel(2, 8, 64, 32, arrival=10.0)]
    run(ks, ContentionILP(ks, override=override, ilp=False))
    assert ks[1].resource == accelerator
    assert ks[2].resource == where
    if where == cpu:
        assert ks[2].start == pytest.approx(10.0)


#--- rollout (3) ---

#at t = 100 kernel 2 arrives, the accelerator is busy and a core is free. by standalone times the cpu
#finishes it 3 ns sooner, so the greedy starts it there. but on the cpu it fights the accelerator for
#dram and both slow down. rollout plays both out, sees that, and waits for the accelerator
def test_rollout_waits_instead_of_contending():
    def make():
        return [make_gmm_kernel(0, 16, 64, 16, arrival=0.0),
                make_gmm_kernel(1, 32, 32, 64, arrival=0.0),
                make_gmm_kernel(2, 16, 32, 16, arrival=100.0)]
    ks = make()
    greedy = run(ks, ContentionILP(ks, ilp=False, adjusted=False))
    assert ks[2].resource == cpu
    ks = make()
    rolled = run(ks, ContentionILP(ks, ilp=False, adjusted=False, rollout_k=1))
    assert ks[2].resource == accelerator
    assert makespan(rolled) < makespan(greedy)


#the greedy's own pick is always an option and wins ties => never worse than the greedy
@pytest.mark.parametrize("k", [1, 2, "full"])
@pytest.mark.parametrize("make", families)
def test_rollout_never_worse(make, k):
    ks = make()
    greedy = makespan(run(ks, ContentionILP(ks)))
    ks = make()
    policy = ContentionILP(ks, rollout_k=k)
    sim = run(ks, policy)
    assert makespan(sim) <= greedy + 1e-6
    assert policy.n_rollouts > 0
    by_id = {k.kernel_id: k for k in ks}
    assert all(k.start >= by_id[d].finish for k in ks for d in k.deps)


#rollout_k = 0 is the plain greedy: same schedule, nothing played out
def test_rollout_zero_is_greedy():
    ks = random_dag(seed=0)
    run(ks, ContentionILP(ks))
    plain = [(k.kernel_id, k.resource, k.start) for k in ks]
    ks = random_dag(seed=0)
    policy = ContentionILP(ks, rollout_k=0)
    run(ks, policy)
    assert [(k.kernel_id, k.resource, k.start) for k in ks] == plain
    assert policy.n_rollouts == 0


def test_bad_rollout_k():
    with pytest.raises(ValueError):
        ContentionILP(random_dag(seed=0), rollout_k=-1)

#--- heuristics (4): malleable cores ---

#compute bound: 100 ops per element => 10000 ns of compute, 4000 bytes = 400 ns of dram
def compute_bound(kid, arrival=0.0, deps=()):
    return make_simple_kernel(kid, 1000, arrival=arrival, ops_per_element=100, deps=deps)


#alone with 2 free cores it takes both: speedup 1 + 0.5 => 10000 / 1.5 ns
@pytest.mark.parametrize("malleable, n, finish", [(True, 2, 10000 / 1.5), (False, 1, 10000.0)])
def test_malleable_takes_the_spare_core(malleable, n, finish):
    ks = [compute_bound(0)]
    sim = run(ks, ContentionILP(ks, malleable=malleable))
    assert sim.runs[0].num_cores == n
    assert ks[0].finish == pytest.approx(finish)


#memory bound: 100 ns of compute but 400 ns of dram, a 2nd core changes nothing => it stays on 1
def test_malleable_skips_memory_bound():
    ks = [make_simple_kernel(0, 1000, arrival=0.0)]
    sim = run(ks, ContentionILP(ks, malleable=True))
    assert sim.runs[0].num_cores == 1


#2 ready kernels, 2 cores: no core is spare, each gets 1
def test_malleable_never_takes_a_needed_core():
    ks = [compute_bound(0), compute_bound(1)]
    sim = run(ks, ContentionILP(ks, malleable=True))
    assert sim.runs[0].num_cores == sim.runs[1].num_cores == 1
    assert ks[0].start == ks[1].start == 0.0


#busy_until counts the cores the running kernel got: 2 cores => it ends at 10000 / 1.5, not 10000
def test_busy_until_counts_the_cores():
    ks = [compute_bound(0)]
    policy = ContentionILP(ks, malleable=True)
    sim = Simulator(make_system(), policy)
    sim.submit(ks)
    sim.run(stop=100.0)
    assert policy.busy_until(ks[0], sim) == pytest.approx(10000 / 1.5)


#rollout also tries the pick without the extra cores, and still never loses to its own greedy
@pytest.mark.parametrize("make", families)
def test_malleable_rollout_never_worse(make):
    ks = make()
    greedy = makespan(run(ks, ContentionILP(ks, malleable=True)))
    ks = make()
    sim = run(ks, ContentionILP(ks, malleable=True, rollout_k=1))
    assert makespan(sim) <= greedy + 1e-6
    by_id = {k.kernel_id: k for k in ks}
    assert all(k.start >= by_id[d].finish for k in ks for d in k.deps)

#--- heuristics (4): pure functions ---

#kernel 0 is the last unfinished dep of gemm 2 (planned on the accelerator) => its accelerator time.
#while kernel 1 still runs it unlocks nothing, and a cpu-planned child never counts
def test_feed_bonus():
    children, deps = {0: [2, 3]}, {2: (0, 1), 3: (0,)}
    plan = {2: accelerator, 3: cpu}
    cost = {2: {cpu: 900.0, accelerator: 300.0}, 3: {cpu: 50.0}}
    assert feed_bonus(0, children, deps, plan, cost, done={1}) == 300.0
    assert feed_bonus(0, children, deps, plan, cost, done=set()) == 0.0


#under the peak nobody loses anything. over it a candidate loses its time x the share dram cant serve,
#so a compute bound one (small demand) loses less than a memory bound one
def test_pairing_penalty():
    assert pairing_penalty(100.0, 4e9, 5e9, 10e9) == 0.0
    assert pairing_penalty(100.0, 0.0, 50e9, 10e9) == 0.0
    assert pairing_penalty(100.0, 10e9, 10e9, 10e9) == pytest.approx(50.0)
    assert pairing_penalty(100.0, 1e9, 10e9, 10e9) < pairing_penalty(100.0, 10e9, 10e9, 10e9)


#the same 100 ns of harm delays a kernel if it lands on the critical one (rank = top), not if it lands
#on one with little behind it. waiting 50 ns costs a kernel at half the top rank 25
def test_should_delay():
    rank = {0: 1000.0, 1: 100.0}
    assert should_delay({0: 100.0}, rank, 1000.0, wait_ns=50.0, my_rank=500.0)
    assert not should_delay({1: 100.0}, rank, 1000.0, wait_ns=50.0, my_rank=500.0)


def test_contended():
    assert contended([6e9, 5e9], 10e9)
    assert not contended([6e9, 4e9], 10e9)


#--- heuristics (4): in the policy ---

#pdf 2.5: k0 memory bound heads k0 -> k2 -> k3, k1 memory bound and short. without delay k1 slows k0 and
#the whole chain (148.4 us). with delay k1 waits for k0 and runs next to compute bound k2 (140.4 us)
def delay_chain():
    return [make_simple_kernel(0, 100000, arrival=0.0), make_simple_kernel(1, 20000, arrival=0.0),
            make_simple_kernel(2, 1000, arrival=0.0, ops_per_element=1000, deps=(0,)),
            make_simple_kernel(3, 1000, arrival=0.0, deps=(2,))]


@pytest.mark.parametrize("delay, k1_start, end", [(False, 0.0, 148400.0), (True, 40000.0, 140432.0)])
def test_delay_waits_for_the_chain(delay, k1_start, end):
    ks = delay_chain()
    policy = ContentionILP(ks, delay=delay)
    sim = run(ks, policy)
    assert ks[1].start == pytest.approx(k1_start)
    assert makespan(sim) == pytest.approx(end)
    assert policy.n_delays == (1 if delay else 0)


#with a huge bandwidth dram only fills up during the accelerator's pure load phases (no compute =>
#infinite demand): the trigger skips most delay checks and the run stays the same
def test_trigger_skips_when_dram_fits():
    config = SystemConfig(cores=make_system().cores, accelerator=make_system().accelerator,
                          memory=MemoryConfig(dram_bw_bytes_per_s=1e15))
    ks = random_dag(seed=0)
    always = ContentionILP(ks, delay=True)
    end = makespan(run(ks, always, config))
    ks = random_dag(seed=0)
    gated = ContentionILP(ks, delay=True, trigger=True)
    assert makespan(run(ks, gated, config)) == pytest.approx(end)
    assert gated.n_checks < always.n_checks / 2


#1 core: k1 (rank 800) beats k0 (rank 400 + its gemm 332.8 = 732.8). with feeding k0 gets the gemm's
#332.8 ns on top (the accelerator is idle), so it starts first and the gemm starts sooner
@pytest.mark.parametrize("feed, first", [(False, 1), (True, 0)])
def test_feed_starts_the_feeder_first(feed, first):
    ks = [make_simple_kernel(0, 1000, arrival=0.0), make_simple_kernel(1, 1000, arrival=0.0),
          make_gmm_kernel(2, 10, 32, 32, arrival=0.0, deps=(0,)), make_simple_kernel(3, 1000, arrival=0.0, deps=(1,))]
    run(ks, ContentionILP(ks, feed=feed), make_system(n_cores=1))
    assert ks[first].start == 0.0
    assert ks[2].resource == accelerator


#every heuristic finishes every family with the deps respected, alone and all together
@pytest.mark.parametrize("flags", [dict(harm=True), dict(pairing=True), dict(feed=True), dict(delay=True),
                                   dict(delay=True, trigger=True),
                                   dict(malleable=True, harm=True, pairing=True, feed=True, delay=True, trigger=True)])
@pytest.mark.parametrize("make", families)
def test_heuristics_finish_everything(make, flags):
    ks = make()
    run(ks, ContentionILP(ks, **flags))
    by_id = {k.kernel_id: k for k in ks}
    assert all(k.done for k in ks)
    assert all(k.start >= by_id[d].finish for k in ks for d in k.deps)


#--- re-plan (5) ---

def run_noisy(kernels, policy, **noise):
    sim = Simulator(make_system(), policy, **noise)
    sim.submit(kernels)
    sim.run()
    return sim


#a kernel alone with an exact model: actual = predicted, the drift stays 1 and nothing re-plans
def test_drift_exact_model():
    ks = [compute_bound(0)]
    policy = ContentionILP(ks, replan=True)
    run_noisy(ks, policy)
    assert policy.drift == {cpu: pytest.approx(1.0), accelerator: 1.0}
    assert policy.n_replans == 0


#cpu kernels really take 2x: a chain of 3 alone, each moves the drift a quarter of the way to 2
#(measured against the uncorrected prediction, so a re-plan in between doesnt change it)
def test_drift_follows_the_bias():
    ks = [make_simple_kernel(i, 1000, arrival=0.0, deps=(i - 1,) if i else ()) for i in range(3)]
    policy = ContentionILP(ks, replan=True)
    run_noisy(ks, policy, bias={cpu: 2.0})
    assert policy.drift[cpu] == pytest.approx(2 - 0.75 ** 3)


#gemm chain, 168.4 ns on the accelerator vs 204.8 on a core. the accelerator really takes 2x: after the
#1st gemm the drift is 1.25 > 15%, the re-plan puts the next ones on the cpu (210.5 > 204.8)
@pytest.mark.parametrize("replan, end", [(False, 3 * 336.8), (True, 336.8 + 2 * 204.8)])
def test_replan_moves_the_next_kernels(replan, end):
    ks = [make_gmm_kernel(i, 2, 16, 32, arrival=0.0, deps=(i - 1,) if i else ()) for i in range(3)]
    policy = ContentionILP(ks, replan=replan)
    sim = run_noisy(ks, policy, bias={accelerator: 2.0})
    assert makespan(sim) == pytest.approx(end)
    assert policy.n_replans == (1 if replan else 0)
    if replan:
        assert [k.resource for k in ks] == [accelerator, cpu, cpu]
        assert policy.cost[1][accelerator] == pytest.approx(policy.alone[1][accelerator] * 1.25)


#threshold = inf never re-plans => the same run as without re-plan, noise or not
@pytest.mark.parametrize("ilp", [False, True])
def test_replan_infinite_threshold(ilp):
    ks = random_dag(seed=0)
    plain = makespan(run_noisy(ks, ContentionILP(ks, ilp=ilp), noise=0.3))
    ks = random_dag(seed=0)
    policy = ContentionILP(ks, ilp=ilp, replan=True, threshold=float("inf"))
    assert makespan(run_noisy(ks, policy, noise=0.3)) == pytest.approx(plain)
    assert policy.n_replans == 0


#the ilp re-plan of the kernels not started yet keeps every dep, with noise and a bias
def test_ilp_replan_finishes_everything():
    ks = fork_join(seed=0)
    policy = ContentionILP(ks, ilp=True, replan=True)
    run_noisy(ks, policy, noise=0.3, bias={accelerator: 1.5})
    by_id = {k.kernel_id: k for k in ks}
    assert all(k.done for k in ks) and policy.n_replans > 0
    assert all(k.start >= by_id[d].finish for k in ks for d in k.deps)


#rollout's copy runs without noise: playing out from before anything starts on a noisy simulator gives
#the noiseless makespan, not the real one (the policy only knows the predictions)
def test_rollout_copy_has_no_noise():
    def fresh_sim(**noise):
        ks = random_dag(seed=0)
        policy = ContentionILP(ks)
        sim = Simulator(make_system(), policy, **noise)
        sim.submit(ks)
        return sim, policy
    noisy, policy = fresh_sim(noise=0.5, bias={cpu: 1.7})
    shared = (policy.cost, policy.rank, policy.plan, policy.n_desc, noisy.config)
    played = play_out(noisy, [], shared) #nothing started yet, the copy runs the whole dag
    exact, _ = fresh_sim()
    assert played == pytest.approx(exact.run())
    assert played != pytest.approx(noisy.run())


@pytest.mark.parametrize("bad", [dict(lam=-1.0), dict(threshold=0.0), dict(alpha=0.0), dict(alpha=1.5)])
def test_bad_heuristic_args(bad):
    with pytest.raises(ValueError):
        ContentionILP(random_dag(seed=0), **bad)


#the presets only switch on flags that exist, and both finish a dag
@pytest.mark.parametrize("preset", [best, best_rollout])
def test_presets(preset):
    ks = fork_join(seed=0)
    run(ks, ContentionILP(ks, **preset))
    assert all(k.done for k in ks)