"""
gchill, the new algorithm: offline plan + online greedy dispatch, on the same 2 phase skeleton as
eshcs (xu 2023) so every difference can be measured on its own

offline, once, on the 1st arrival (prepare), needs the whole dag:
- cost: each kernel's time alone on each resource (costs.py), adjusted=True for the dram sharing version
- plan: which resource each kernel goes on (plan.py), fastest alone by default, ilp=True for the whole-dag ilp
- rank: upward rank by default (priority.py), the longest chain of planned time from a kernel to the end
  => a high rank means a lot of work waits behind it, so it goes first

online, after every arrival and finish (pick):
- greedy: the ready kernels in rank order, each books its planned resource (later ones see the queue)
- override: if the other resource is free and finishes the kernel sooner than waiting, start it there
- rollout_k > 0: play each option out on a copy of the simulator, keep the one that ends the dag first (rollout.py)

heuristics (4), flags, the logic is in heuristics.py:
- malleable: the cores left free go to the cpu picks that gain the most from them
- harm, pairing, feed: change the order of the ready kernels (rank - harm, rank - pairing loss, rank + feeding)
- delay: hold a kernel back if it would slow the critical running kernels more than waiting costs it
- trigger: the delay check only runs when dram is oversubscribed
re-plan (5): replan=True corrects the costs by the observed error and re-plans the kernels not started

defaults = fastest-alone plan, standalone costs, upward rank, override on, no rollout, no heuristics
(the e4 base). the final configs are the presets below: ContentionILP(kernels, **best)
"""

import time
from sim.kernels import cpu, accelerator, topological
from sim.estimate import standalone_ns
from sim.policies.base import Policy
from sim.policies.cost_model import what_if, segments
from sim.policies.gchill.costs import cost_table, adjusted_cost_table
from sim.policies.gchill.plan import core_plan, ilp_plan
from sim.policies.gchill.priority import descendants, PRIORITY
from sim.policies.gchill.rollout import options, play_out
from sim.policies.gchill.heuristics import feed_bonus, pairing_penalty, should_delay, contended

min_gain = 0.01 #an extra core has to cut the kernel's time by at least 1% (a memory bound one gains ~0)

#the final configs (e4): the best greedy (cheap, the dispatch-unit candidate) and the best overall
#(+ rollout, software only). ilp, adjusted, harm, pairing, feed lost and replan only pays off when the
#model is off the same way for a whole resource, so they stay off
best = dict(malleable=True, delay=True, trigger=True)
best_rollout = dict(best, rollout_k=3)


class ContentionILP(Policy):
    def __init__(self, kernels, override=True, priority="upward", adjusted=False, ilp=False, rollout_k=0,
                 malleable=False, harm=False, lam=1.0, pairing=False, feed=False, delay=False, trigger=False,
                 replan=False, threshold=0.15, alpha=0.25):

        if priority not in PRIORITY:
            raise ValueError(f"Unknown priority: {priority}, pick one of {list(PRIORITY)}")
        if rollout_k != "full" and not (isinstance(rollout_k, int) and rollout_k >= 0):
            raise ValueError(f"rollout_k: {rollout_k} must be an int >= 0 or 'full'")
        if lam < 0 or threshold <= 0 or not 0 < alpha <= 1:
            raise ValueError(f"lam: {lam} must be >= 0, threshold: {threshold} > 0, alpha: {alpha} in (0, 1]")

        self.kernels = topological(kernels)
        self.override = override
        self.priority = priority
        self.cost = None #{kernel id: {resource: ns}}
        self.plan = None #{kernel id: cpu or accelerator}, made on the 1st on_queued
        self.rank = None #{kernel id: upward rank in ns}
        self.n_desc = None #how many kernels come after it, break ties in rank
        self.ready = [] #released, not started yet
        self.adjusted = adjusted
        self.ilp = ilp #true = ilp plan, else fastest alone (core plan)
        self.T = None #ilp estimated makespan, none without the ilp
        self.solve_ms = None #how long making the plan took (one time cost)
        self.rollout_k = rollout_k #0 = plain greedy, k = try the top k ready kernels, "full" = all of them
        self.n_rollouts = 0 #how many options were played out, the cost of rollout

        #(4) heuristics, one flag each so the e4 ablation turns them on one at a time
        self.malleable = malleable #the cores left free after a pick go to the cpu picks that gain the most
        self.harm, self.lam = harm, lam #order by rank - lam x the ns a kernel adds to the running ones
        self.pairing = pairing #order by rank - the ns a kernel would lose to dram it cannot get
        self.feed = feed #order by rank + the accelerator work a cpu kernel unlocks, if the accelerator would idle
        self.delay = delay #hold a kernel back if it hurts critical running kernels more than waiting costs
        self.trigger = trigger #the delay check only runs when dram is oversubscribed
        self.n_delays = 0 #kernels held back
        self.n_checks = 0 #delay checks that ran (the costly what_if replay)

        #(5) re-plan
        self.replan, self.threshold, self.alpha = replan, threshold, alpha
        self.drift = {cpu: 1.0, accelerator: 1.0} #ewma of actual / predicted time per resource
        self.basis = dict(self.drift) #the drift the current costs were corrected with
        self.started = [] #kernels started and not seen finishing yet
        self.expected = {} #{kernel id: ns the cost model expected when it started}
        self.n_replans = 0
        self.replan_ms = 0.0


    #offline phase, runs once. needs sim for the hardware, so it waits for the 1st arrival
    def on_queued(self, kernel, sim):
        if self.plan is None:
            self.prepare(sim)
        self.ready.append(kernel)

    #ns on n cores alone. 1 core = the cost table
    def cpu_ns(self, kernel, n, sim):
        if n == 1:
            return self.cost[kernel.kernel_id][cpu]
        return standalone_ns(kernel, cpu, sim, n) * self.cost[kernel.kernel_id][cpu] / self.alone[kernel.kernel_id][cpu]

    #when this resource is next free: now if idle, else when the kernel on it should finish
    #(on as many cores as it got)
    def busy_until(self, kernel, sim):
        if kernel.resource == cpu:
            took = self.cpu_ns(kernel, sim.runs[kernel.kernel_id].num_cores, sim)
        else:
            took = self.cost[kernel.kernel_id][accelerator]
        return max(sim.now, kernel.start + took)

    #where this kernel goes: its planned resource, unless the other one is free right now and
    #finishes it sooner. finish = when the resource is free (counting the kernels ahead of it in
    #this pick) + its time there
    def choose(self, kernel, free_at, free):
        c = self.cost[kernel.kernel_id]
        planned = self.plan[kernel.kernel_id]
        finish = {cpu: free_at[cpu][0] + c[cpu]}
        if accelerator in c: #only kernels that fit the accelerator have a time there
            finish[accelerator] = free_at[accelerator] + c[accelerator]
        other = cpu if planned == accelerator else accelerator
        if self.override and free[other] and other in finish and finish[other] < finish[planned]:
            return other, finish[other]
        return planned, finish[planned]

    #sort key: highest rank first, then the one with more kernels after it, then the lowest id
    def order(self, kernel):
        kid = kernel.kernel_id
        return (-self.rank[kid], -self.n_desc[kid], kid)

    #the ready kernels in the order the greedy tries them: rank order, or with harm / pairing / feed
    #the rank moved by their ns (they all stack)
    def queue(self, sim, free_at):
        if not (self.harm or self.pairing or self.feed):
            return sorted(self.ready, key=self.order)
        score = {k.kernel_id: self.rank[k.kernel_id] for k in self.ready}
        if self.harm and sim.jobs:
            for k in self.ready:
                score[k.kernel_id] -= self.lam * what_if(k, self.plan[k.kernel_id], sim).harm
        if self.pairing:
            running = sum(j.demand() for j in sim.jobs.values())
            peak = sim.config.memory.dram_bw_bytes_per_s
            for k in self.ready:
                kid, where = k.kernel_id, self.plan[k.kernel_id]
                demand = segments(k, where, sim)[0].demand()
                score[kid] -= pairing_penalty(self.cost[kid][where], demand, running, peak)
        if self.feed:
            done = set(sim.jobs) | {k.kernel_id for k in sim.kernels if k.done}
            for k in self.ready:
                kid = k.kernel_id
                #only if the accelerator would sit idle before this kernel is done on a core
                if self.plan[kid] == cpu and free_at[accelerator] <= free_at[cpu][0] + self.cost[kid][cpu]:
                    score[kid] += feed_bonus(kid, self.children, self.deps, self.plan, self.cost, done)
        return sorted(self.ready, key=lambda k: (-score[k.kernel_id], -self.n_desc[k.kernel_id], k.kernel_id))

    #delay: would starting this kernel now hurt the critical running kernels more than waiting till the
    #next event costs it? only if something runs or was just picked, so the next event exists (no deadlock)
    def held(self, kernel, kind, sim, picks):
        ends = [self.busy_until(sim.by_id[kid], sim) for kid in sim.jobs]
        ends += [sim.now + (self.cpu_ns(k, n, sim) if where == cpu else self.cost[k.kernel_id][accelerator])
                 for k, where, n in picks] #picked in this same pick, they start now
        wait = min(ends, default=sim.now) - sim.now
        if wait <= 0:
            return False
        if self.trigger:
            demands = [j.demand() for j in sim.jobs.values()]
            demands += [segments(k, where, sim, n or 1)[0].demand() for k, where, n in picks]
            demands.append(segments(kernel, kind, sim)[0].demand())
            if not contended(demands, sim.config.memory.dram_bw_bytes_per_s):
                return False
        self.n_checks += 1
        harm_by = what_if(kernel, kind, sim, starting=picks).harm_by
        if should_delay(harm_by, self.rank, max(self.rank.values()), wait, self.rank[kernel.kernel_id]):
            self.n_delays += 1
            return True
        return False

    #the ready kernels in order. each one books its resource, even if it has to wait, so the kernels after
    #it see the queue it leaves. first = (kernel, kind) to start before everyone else (rollout).
    #only returns the picks, self.ready doesnt change. widen=False = no extra cores (a rollout option)
    def greedy(self, sim, first=None, widen=True):
        picks = []
        free_cores = len(sim.cpu.free_cores())
        free = {cpu: free_cores > 0, accelerator: sim.accelerator.free}
        free_at = { #when each core (soonest first) and the accelerator are free
            cpu: sorted(sim.now if c.free else self.busy_until(c.current, sim) for c in sim.cpu.cores),
            accelerator: sim.now if free[accelerator] else self.busy_until(sim.accelerator.current, sim),
        }
        queue = self.queue(sim, free_at)
        if first is not None:
            queue.remove(first[0])
            queue.insert(0, first[0])

        for kernel in queue:
            if not free[cpu] and not free[accelerator]:
                break
            forced = first is not None and kernel is first[0]
            if forced:
                kind = first[1]
                start = free_at[cpu][0] if kind == cpu else free_at[accelerator]
                finish = start + self.cost[kernel.kernel_id][kind]
            else:
                kind, finish = self.choose(kernel, free_at, free)
            if self.delay and not forced and free[kind] and self.held(kernel, kind, sim, picks):
                continue #it waits for the next event, nothing booked
            if kind == cpu:
                free_at[cpu][0] = finish #the core that frees up first is booked till then
                free_at[cpu].sort()
            else:
                free_at[accelerator] = finish
            if not free[kind]:
                continue #it waits, but its booking stays
            if kind == accelerator:
                picks.append((kernel, accelerator, None))
                free[accelerator] = False
            else:
                picks.append((kernel, cpu, 1))
                free_cores -= 1
                free[cpu] = free_cores > 0

        if self.malleable and widen and free_cores > 0:
            picks = self.widen(picks, free_cores, sim)
        return picks

    #malleable cores: every ready kernel is placed or waits, so the free cores go 1 at a time to the cpu
    #pick whose time drops the most with 1 more core. a kernel keeps its cores till it finishes
    def widen(self, picks, spare, sim):
        picks = [list(p) for p in picks]
        while spare > 0:
            chosen, most = None, 0.0
            for p in picks:
                kernel, kind, n = p
                if kind != cpu or not kernel.parallelize:
                    continue
                now = self.cpu_ns(kernel, n, sim)
                drop = now - self.cpu_ns(kernel, n + 1, sim)
                if drop > min_gain * now and drop > most:
                    chosen, most = p, drop
            if chosen is None:
                break
            chosen[2] += 1
            spare -= 1
        return [tuple(p) for p in picks]

    #drift: each kernel that finished since the last pick moves its resource's error toward actual /
    #expected, expected = the cost model's time when it started, with the co-runners known then. so the
    #drift is the model's error, not the dram sharing (the greedy already reacts to that)
    def track(self, sim):
        for k in [k for k in self.started if k.done]:
            self.started.remove(k)
            ratio = (k.finish - k.start) / self.expected.pop(k.kernel_id)
            self.drift[k.resource] += self.alpha * (ratio - self.drift[k.resource])

    #re-plan when a resource's error moved more than threshold since the last correction: every cost x
    #its resource's drift, a new plan for the kernels not started (released after their deps), new ranks
    def maybe_replan(self, sim):
        if not any(abs(self.drift[r] / self.basis[r] - 1) > self.threshold for r in self.drift):
            return
        t0 = time.perf_counter()
        self.basis = dict(self.drift)
        #a new dict, not an in-place change: rollout's copies share the old one
        self.cost = {kid: {r: t * self.drift[r] for r, t in c.items()} for kid, c in self.alone.items()}
        waiting = [k for k in self.kernels if k.start is None]
        if self.ilp and waiting:
            ids = {k.kernel_id for k in waiting}
            release = {} #a waiting kernel cant start before its done or running deps finish
            for k in waiting:
                ends = [sim.by_id[d].finish if sim.by_id[d].done else self.busy_until(sim.by_id[d], sim)
                        for d in k.deps if d not in ids]
                release[k.kernel_id] = max([sim.now] + ends)
            new, self.T = ilp_plan(waiting, self.cost, sim, release=release)
        else:
            new = core_plan(waiting, self.cost)
        self.plan = {**self.plan, **new}
        self.rank = PRIORITY[self.priority](self.kernels, self.plan, self.cost, sim)
        self.n_replans += 1
        self.replan_ms += (time.perf_counter() - t0) * 1e3

    #the greedy's picks, or with rollout the option that finishes the dag first when played out
    def pick(self, sim):
        if self.replan:
            self.track(sim)
            self.maybe_replan(sim)
        if self.rollout_k == 0:
            picks = self.greedy(sim)
        else:
            found = options(self, sim, self.rollout_k)
            picks = found[0]
            if len(found) > 1:
                shared = (self.cost, self.rank, self.plan, self.n_desc, sim.config)
                self.n_rollouts += len(found)
                picks = min(found, key=lambda p: play_out(sim, p, shared)) #ties => the 1st = greedy
        for kernel, kind, n in picks:
            self.ready.remove(kernel)
            if self.replan:
                others = [p for p in picks if p[0] is not kernel]
                self.expected[kernel.kernel_id] = what_if(kernel, kind, sim, n or 1, starting=others).time
                self.started.append(kernel)
        return picks

    def prepare(self, sim):
        table = adjusted_cost_table if self.adjusted else cost_table
        self.cost = table(self.kernels, sim)
        self.alone = self.cost #the predictions before any re-plan correction
        start = time.perf_counter()
        if self.ilp:
            self.plan, self.T = ilp_plan(self.kernels, self.cost, sim)
        else:
            self.plan = core_plan(self.kernels, self.cost)
        self.solve_ms = (time.perf_counter() - start) * 1e3
        self.rank = PRIORITY[self.priority](self.kernels, self.plan, self.cost, sim)
        self.n_desc = {kid: len(d) for kid, d in descendants(self.kernels).items()}
        self.deps = {k.kernel_id: k.deps for k in self.kernels}
        self.children = {k.kernel_id: [] for k in self.kernels}
        for k in self.kernels:
            for d in k.deps:
                self.children[d].append(k.kernel_id)