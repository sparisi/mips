# Description
-------------

**Mi**nimal **P**olicy **S**earch is a toolbox for Matlab providing the implementation of some of the most famous policy search algorithms, as well as some recent multi-objective methods and benchmark problems in reinforcement learning.

It requires the *Optimization Toolbox*.

Some utility functions are imported from File Exchange (original authors are always acknowledged).


# Code Structure
----------------

Launch `INSTALL` to add the path of all folders.

### Algs
All the algorithms and solvers are located in this folder, as well as some script to run them. By using scripts, it is possible to interrupt and resume the learning process without losing any data.
The only parameters that you might want to change are the learning rates and the number of rollouts per iteration.
Also, a history of the results is usually kept. For example, `J_history` stores the average return at each iteration.

### Experiments
This folder contains some scripts to set up experiments. Each script inizializes the MDP, the policies and the number of samples and episodes per learning iteration.
After running a setup script, just run an algorithm script to start the learning.

Notice that, in the case of episodic (black box) RL, these scripts define both the *low level policy* (the one used by the agent) and the *high level policy* (the sampling distribution used to draw the low level policy parameters).
In this setting, it is important to set up the variable `makeDet`: if `true`, the low level policy is deterministic (e.g., the covariance of a Gaussian is zeroed and the high level policy only draws its mean).

### Library
The folder contains some policies, generic basis functions, and functions for sampling and evaluation. The most important functions are

- `collect_samples`, which stores low level tuples `(s,a,r,s')` into a struct,
- `collect_episodes`, which collects high level data, i.e. pairs `(return,policy)`,
- `evaluate_policies`, which evaluates low level policies on several episodes.

Policies are modeled as objects. Their most important method is `drawAction`, but depending on the class some additional properties might be mandatory.

> **IMPORTANT!** All data is stored in **COLUMNS**, e.g., states are matrices `S x N`, where `S` is the size of one state and `N` is the number of states. Similarly, actions are matrices `A x N` and features are matrices `F x N`.

### MDPs
Each MDP is modeled as an object (`MDP.m`) and requires some properties (dimension of state and action spaces, bounds, etc...) and methods (for simulating and plotting).
There are also some extension, that are *Contextual MDPs* (`CMDP.m`) and *Multi-objective MDPs* (`MOMDP.m`).

### MO_Library
This folder contains functions used in the multi-objective framework, e.g., hypervolume estimators and Pareto-optimality filters.

> **IMPORTANT!** All frontiers is stored in **ROWS**, i.e., they are matrices `N x R`, where `N` is the number of points and `R` is the number of objectives.

### Utilities
Utility functions used for matrix operations, plotting and sampling are stored in this folder.
