"""
heuristics (4) for the new algorithm: pure functions, the policy (ilp.py) calls them behind its flags
- feed_bonus: accelerator feeding (feed)
- pairing_penalty: compute/memory pairing (pairing)
- should_delay: deliberate delay weighted by criticality (delay)
- contended: the contention trigger (trigger)
malleable cores and harm live in ilp.py, they need the policy's cost table / the cost model directly
"""

from sim.kernels import accelerator


#accelerator feeding: ns of accelerator work this cpu kernel unlocks, the accelerator-planned
#children whose only unfinished dep is this kernel. done = kernel ids already finished or running
def feed_bonus(kid, children, deps, plan, cost, done):
    unlocked = 0.0
    for c in children[kid]:
        if plan[c] == accelerator and all(d == kid or d in done for d in deps[c]):
            unlocked += cost[c][accelerator]
    return unlocked


#compute/memory pairing: ns a candidate would lose to dram sharing if it starts next to what runs now,
#its time x the fraction of all the demand dram cannot serve. 0 when it fits under the peak, so a
#compute bound candidate (tiny demand) beats a memory bound one when dram is already saturated
def pairing_penalty(time_ns, demand, running_demand, peak):
    total = demand + running_demand
    if total <= peak or demand == 0:
        return 0.0
    return time_ns * (total - peak) / total


#deliberate delay: start nothing now? harm only lengthens the makespan through critical kernels,
#so each running kernel's extra ns is weighted by its criticality (rank / longest rank, 1 = on the
#longest remaining chain). waiting costs the time to the next event, weighted the same way
def should_delay(harm_by, rank, top, wait_ns, my_rank):
    weighted = sum(extra * rank[kid] / top for kid, extra in harm_by.items())
    return weighted > wait_ns * my_rank / top


#contention trigger: the costly heuristics only run when dram is oversubscribed, i.e. the demand of
#everything running + the candidate is more than the peak bandwidth. otherwise nobody slows anybody
def contended(demands, peak):
    return sum(demands) > peak