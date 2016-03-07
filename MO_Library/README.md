**Normalization in Pareto-ascent Direction**

In PFA and RA, while checking for the minimal-norm Pareto-ascent direction, the gradients are always normalized (i.e., we normalize the gradients even if their norm is less than 1). If the gradients are not normalized, then the minimial-norm Pareto-ascent direction is expected to be mostly influenced by the elements which have smaller norms. Since gradients with small norm are usually associated with objectives that have already achieved a fair degree of convergence, the utility of considering these directions for a well-balanced correction is questionable. 
This observation points out the necessity of normalizing ALL the gradients when looking for the minimal-norm Pareto-ascent direction (see also *Desideri, 'Multiple-gradient descent algorithm for multiobjective optimization', 2012*).

___
**Policy Randomization in PFA**

PFA (usually) needs to randomization the policy after the optimization of each objective. Such randomization is needed to guarantee enough exploration to optimize the remaining objectives and depends on the policy used. 

For instance: 
- for a Gibbs policy we can reduce the temperature or scale theta with a constant factor (e.g., we can halve theta),
- for a Gaussian policy we can reset the covariance matrix at its initial value or we can, again, scale it.