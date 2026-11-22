"""
textbook heft (topcuoglu, hariri, wu 2002), planned once before anything runs:

1. upward rank: rank(k) = mean cost of k + the largest rank of its children. the mean is over every
   processor that can run k (n cpu cores + the accelerator for a gemm, only the cores otherwise).
   communication costs are 0: cpu and accelerator share the dram, nothing is copied between them
2. kernels in rank order (highest first), each goes on the processor where it finishes earliest,
   allowed to fill an idle gap between two kernels already planned there (insertion policy)
3. at run time every processor runs its own list in the planned order: a kernel starts when its
   processor is idle, it was released (arrived + deps done), and everything before it on that
   processor started. heft never reorders or moves a kernel once planned

costs are standalone (1 core, the full dram bandwidth): heft is blind to contention, the plan's
times are wrong when kernels share the dram, only its placement and order are kept.
the cpu cores are identical, so a "processor" here is a virtual core: each one runs 1 kernel at a
time, and there are as many as real cores, so a real core is always free when a virtual one is
"""

from sim.kernels import cpu, accelerator
from sim.policies.base import Policy
from sim.policies.cost_model import segments


class Heft(Policy):
    def __init__(self):
        self.lists = None #{processor: [kernel ids in planned start order]}
        self.where = None #{kernel id: processor}, processor = (cpu, i) or (accelerator, 0)
        self.rank = None #{kernel id: upward rank in ns}
        self.planned_finish = None #{kernel id: planned finish in ns}
        self.released = set() #kernel ids released by the simulator, not started yet
        self.running = {} #{processor: kernel running on it}
        self.next = None #{processor: index of the next kernel in its list}

    #standalone ns of kernel on a resource kind: its segments at the full dram bandwidth
    def cost(self, kernel, kind, sim):
        peak = sim.config.memory.dram_bw_bytes_per_s
        return sum(s.duration_ns(peak) for s in segments(kernel, kind, sim))

    def on_submit(self, kernels, sim):
        self.plan(sim.kernels, sim)

    def plan(self, kernels, sim):
        by_id = {k.kernel_id: k for k in kernels}
        procs = [(cpu, i) for i in range(sim.config.cores.n_cores)] + [(accelerator, 0)]
        w = {} #{kernel id: {processor: ns}}
        for k in kernels:
            w[k.kernel_id] = {p: self.cost(k, p[0], sim) for p in procs
                              if p[0] == cpu or self.fits_accel(k, sim)}

        children = {kid: [] for kid in by_id}
        for k in kernels:
            for d in k.deps:
                children[d].append(k.kernel_id)

        #upward rank, children first (reverse topological order, no recursion)
        order = topo(kernels)
        self.rank = {}
        for kid in reversed(order):
            mean = sum(w[kid].values()) / len(w[kid])
            self.rank[kid] = mean + max((self.rank[c] for c in children[kid]), default=0.0)

        #rank order is a topological order: a parent's rank > its child's (every cost > 0).
        #ties broken by topological position so a parent always comes first
        pos = {kid: i for i, kid in enumerate(order)}
        ranked = sorted(by_id, key=lambda kid: (-self.rank[kid], pos[kid]))

        busy = {p: [] for p in procs} #{processor: sorted [(start, end)]}
        finish = {} #{kernel id: planned finish}
        self.where = {}
        for kid in ranked:
            k = by_id[kid]
            ready = max([finish[d] for d in k.deps] + [k.queue])
            best = None
            for p, t in w[kid].items():
                start = earliest_gap(busy[p], ready, t)
                if best is None or start + t < best[0]:
                    best = (start + t, start, p)
            end, start, p = best
            busy[p].append((start, end))
            busy[p].sort()
            finish[kid] = end
            self.where[kid] = p
        self.planned_finish = finish

        self.lists = {p: [] for p in procs}
        starts = {kid: finish[kid] - w[kid][self.where[kid]] for kid in by_id}
        for kid in sorted(by_id, key=lambda kid: (starts[kid], pos[kid])):
            self.lists[self.where[kid]].append(kid)
        self.next = {p: 0 for p in procs}

    def on_queued(self, kernel, sim):
        if self.lists is None: #submitted without on_submit (should not happen)
            self.plan(sim.kernels, sim)
        self.released.add(kernel.kernel_id)

    def pick(self, sim):
        picks = []
        accel_free = sim.accelerator.free
        for p, kids in self.lists.items():
            running = self.running.get(p)
            if running is not None and not running.done:
                continue #this processor is still busy
            self.running.pop(p, None)
            i = self.next[p]
            if i == len(kids) or kids[i] not in self.released:
                continue #nothing left, or its next kernel is not released yet: keep the order
            if p[0] == accelerator and not accel_free:
                continue
            kernel = sim.by_id[kids[i]]
            self.next[p] += 1
            self.released.discard(kernel.kernel_id)
            self.running[p] = kernel
            picks.append((kernel, p[0], 1 if p[0] == cpu else None))
        return picks


#kernel ids in an order where every dep comes before its children (kahn, stable by kernel id)
def topo(kernels):
    indeg = {k.kernel_id: 0 for k in kernels}
    children = {k.kernel_id: [] for k in kernels}
    for k in kernels:
        for d in k.deps:
            if d in indeg:
                indeg[k.kernel_id] += 1
                children[d].append(k.kernel_id)
    free = sorted(kid for kid, n in indeg.items() if n == 0)
    order = []
    while free:
        kid = free.pop(0)
        order.append(kid)
        for c in children[kid]:
            indeg[c] -= 1
            if indeg[c] == 0:
                free.append(c)
        free.sort()
    return order


#earliest start >= ready where a kernel of length t fits on a processor with these busy slots
#(sorted (start, end)): the first idle gap that is long enough, or after the last slot
def earliest_gap(slots, ready, t):
    start = ready
    for s, e in slots:
        if start + t <= s:
            return start
        start = max(start, e)
    return start
