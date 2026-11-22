"""
rollout (3): look ahead before a pick. each option for this pick is tried on a copy of the
simulator, the plain greedy finishes the whole dag from there, and the option with the shortest
makespan wins. the copy runs the real rate model, so it sees the dram sharing the greedy's
standalone estimates miss

options at a pick:
- the greedy's own pick, always 1st => it wins ties, so rollout is never worse than the greedy
- each of the top k ready kernels (by rank) started first on each free resource it can use,
  then the greedy fills the other free resources
- wait: start nothing now (only if something is running), this is how rollout finds a delay
- with malleable cores: the greedy's pick without the extra cores, so rollout decides if they pay off
"""


import copy
from sim.kernels import cpu, accelerator

#the options for this pick, the greedy's pick 1st, no duplicates
def options(policy, sim, k):
    greedy = policy.greedy(sim)
    found = [greedy]
    if not greedy:
        return found #nothing is free, nothing to decide
    free = {cpu: len(sim.cpu.free_cores()) > 0, accelerator: sim.accelerator.free}
    ready = sorted(policy.ready, key=policy.order)
    top = ready if k == "full" else ready[:k]
    for kernel in top:
        for kind in (cpu, accelerator):
            if free[kind] and kind in policy.cost[kernel.kernel_id]:
                picks = policy.greedy(sim, first=(kernel, kind))
                if picks not in found:
                    found.append(picks)
    if sim.jobs:
        found.append([])

    if policy.malleable:
        narrow = policy.greedy(sim, widen=False)
        if narrow not in found:
            found.append(narrow)
            
    return found

#makespan if these picks start now and the greedy does the rest. shared = things that never change,
#the copy points at them instead of copying them
def play_out(sim, picks, shared):
    twin = copy.deepcopy(sim, {id(x): x for x in shared})
    twin.policy.rollout_k = 0 #the copy finishes with the plain greedy
    for kernel, kind, n in picks:
        k = twin.by_id[kernel.kernel_id]
        twin.policy.ready.remove(k)
        twin.start(k, kind, n)
    twin.reallocate() #what the real event does right after its pick
    twin.run()
    return max(k.finish for k in twin.kernels) - min(k.queue for k in twin.kernels)
