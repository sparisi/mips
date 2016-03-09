# Description
-------------

**Mi**nimal **P**olicy **S**earch is a toolbox for Matlab providing the implementation of some of the most famous policy search algorithms, as well as some recent multi-objective methods and benchmark problems in reinforcement learning.

It requires the *Optimization Toolbox*.

Some utility functions are imported from File Exchange (original authors are always acknowledged).


# Code structure
----------------

Launch `INSTALL` to add the path of all folders.

### Algs
All the algorithms and solvers are located in this folder, as well as some script to run them. By using scripts, it is possible to interrupt and resume the learning process without losing any data.
The only parameters that you might want to change are the learning rates and the number of rollouts per iteration.
Also, a history of the results is usually kept. For example, `J_history` stores the average return at each iteration.

### Experiments
This folder contains some scripts to set up experiments. Each script inizializes the MDP, the policies and the number of samples and episodes per learning iteration.
After running a setup script, just run an algorithm script to start the learning.

`SettingMC % mountain car setup` <br />
`RUN_PG % run policy gradient (terminate by CTRL+C)` <br />
`plot(J_history) % plot average return` <br />
`show_simulation(mdp,policy.makeDeterministic,0.1,1000) % visualize learned policy (see below)`

Notice that, in the case of episodic (black box) RL, these scripts define both the *low level policy* (the one used by the agent) and the *high level policy* (the sampling distribution used to draw the low level policy parameters).
In this setting, it is important to set the variable `makeDet`: if `true`, the low level policy is deterministic (e.g., the covariance of a Gaussian is zeroed and the high level policy only draws its mean).

### Library
The folder contains some policies, generic basis functions, and functions for sampling and evaluation. The most important functions are

- `collect_samples`: stores low level tuples `(s,a,r,s')` into a struct,
- `collect_episodes`: collects high level data, i.e. pairs `(return,policy)`,
- `evaluate_policies`: evaluates low level policies on several episodes,
- `evaluate_policies_high`: evaluates high level policies on several episodes.

Policies are modeled as objects. Their most important method is `drawAction`, but depending on the class some additional properties might be mandatory.

> **IMPORTANT!** All data is stored in **COLUMNS**, e.g., states are matrices `S x N`, where `S` is the size of one state and `N` is the number of states. Similarly, actions are matrices `A x N` and features are matrices `F x N`.

### MDPs
Each MDP is modeled as an object (`MDP.m`) and requires some properties (dimension of state and action spaces, bounds, etc...) and methods (for simulating and plotting).
There are also some extension, that are *Contextual MDPs* (`CMDP.m`) and *Multi-objective MDPs* (`MOMDP.m`).

> **IMPORTANT!** To allow parallel execution of multiple episodes, `simulator` functions need to support vectorized operations, i.e., they need to deal with states and actions represented as `S x N` and `A x N` matrices, repsectively.

### MO_Library
This folder contains functions used in the multi-objective framework, e.g., hypervolume estimators and Pareto-optimality filters.

> **IMPORTANT!** All frontiers are stored in **ROWS**, i.e., they are matrices `N x R`, where `N` is the number of points and `R` is the number of objectives.

### Utilities
Utility functions used for matrix operations, plotting and sampling are stored in this folder.


# Visualization
---------------

Here is a list with examples of all ways of visualizing a particular data / animation. Please note that not all MDPs support an animation.

### Real time data plotting
During the learning, it is possible to plot in real time a desired data (e.g., the return `J`) by using `updateplot`. 

`updateplot('Return',iter,J,1)`

### Mean and std of data from multiple trials
If you are interested on evaluating an algorithm on several trials you can use the function `shadedErrorBar`. For a complete example, please refer to `myplot.m`.

### Real time animation
Launch `mdp.showplot` to initialize the plotting and an animation of the agent-environment interaction will be shown during the learning. To stop plotting use `mdp.closeplot`.

> **IMPORTANT!** This is possible only if you are learning using one episode per iteration.

### Offline animation

- For step-based algorithms, you can directly use the built-in plotting function of the MDPs.
As `collect_samples` returns a low-level dataset of the episodes, you just have to call `mdp.plotepisode`

`data = collect_samples(mdp,policy,episodes,steps,policy)` <br />
`mdp.plotepisode(data(1),0.001)`

- For episode-based algorithms, the low-level dataset is not returned. In this case, you can call `show_simulation`, which executes only one episode and shows an animation. This approach can be used also in step-based algorithms.

`show_simulation(mdp,policy,0.001,100)` <br />
`show_simulation(mdp,policy.update(policy_high.drawAction(1)),0.001,100)`

### MOMDPs Pareto frontier

To plot a set of points as a Pareto frontier of a MOMDP, use `mdp.plotfront`. Each MOMDP has its own plotting style for better clarity. Please note that the points have to be passed as rows and that the function does not filter dominated points.

`mdp.plotfront([0.5 0.5; 1 0; 0 1])`