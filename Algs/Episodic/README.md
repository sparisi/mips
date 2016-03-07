**Episodic RL and Stochasticity**

With episode-based algorithms (e.g., REPS, PGPE, NES) it is recommended to make the lower-level policy deterministic (set `makeDet = 1` in the experiment setting script).
The reason is that such algorithms cannot deal with high stochasticity, as each parameter vector is evaluated only on one episode and therefore its quality estimate would be inaccurate (unless many samples are used to learn). 
For the same reason, these algorithms are not very well-suited for highly stochastic environment.
Anyway, they can manage low stochasticity and if you want your lower-level policy to be stochastic you can of course do it.

___
**Episodic RL and Stabilization**

In episodic policy search (e.g., REPS, PGPE, NES) you can use samples from the last `N_MAX` policies to stabilize the policy update. This stabilization is important to keep the shape of the explorative variance of the policy in high dimensional action spaces (see also *Daniel et al., 'Learning Concurrent Motor Skills in Versatile Solution Spaces', 2012*). However, it also slows the convergence speed down (e.g., the KL divergence in REPS will be 0 only if the last `N_MAX` samples are optimal, i.e., if the variance of the final policy is very low).
