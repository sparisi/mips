**Mi**nimal **P**olicy **S**earch is a toolbox for Matlab providing the implementation of RL algorithms.  
The repository originally focused on policy search (hence the name), especially REPS and policy gradient, but it now contains a wide variety of algrorithms (PPO, TRPO, DQN, DPG, FQI, ...).
It also has multi-objective RL algorithms, benchmark MDPs and optimization problems, and common policies classes. 

Some algorithms require the [Optimization Toolbox](https://www.mathworks.com/products/optimization.html).  
Some utility functions are imported from [File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/) (original authors are always acknowledged).


## Summary

Launch `INSTALL` to add the path of all folders.


<details>
<summary><b>Algs</b></summary>
  <p>
All the algorithms and solvers are located in this folder, as well as some script to run them. By using scripts, it is possible to interrupt and resume the learning process without losing any data.
The only parameters that you might want to change are the learning rates and the number of rollouts per iteration.
Also, a history of the results is usually kept. For example, <code>J_history</code> stores the expected return at each iteration.
  </p>
</details>


<details>
<summary><b>BenchmarkOpt</b></summary>
  <p>
  Here are some <a href="https://en.wikipedia.org/wiki/Test_functions_for_optimization">test functions for optimization</a>.
  </p>
</details>


<details>
<summary><b>Experiments</b></summary>
  <p>
This folder contains some scripts to set up experiments. Each script inizializes the MDP, the policies and the number of samples and episodes per learning iteration.
After running a setup script, just run an algorithm script to start the learning.

```
SettingMC % mountain car setup
RUN_PG % run policy gradient (terminate by CTRL+C)
plot(J_history) % plot expected return
show_simulation(mdp,policy.makeDeterministic,1000,0.1) % visualize learned policy (see below)
```

Notice that, in the case of episodic (black box) RL, these scripts define both the *low level policy* (the one used by the agent) and the *high level policy* (the sampling distribution used to draw the low level policy parameters).
In this setting, the exploration noise is given by the high level policy, while the low level policy is deterministic (e.g., the covariance of a Gaussian is zeroed and the high level policy only draws its mean).
  </p>
</details>


<details>
  <summary><b>Library</b></summary>
  <p>
The folder contains some policies, generic basis functions, and functions for sampling and evaluation. The most important functions are

- `collect_samples`: stores low level tuples `(s,a,r,s')` into a struct,
- `collect_episodes`: collects high level data, i.e. pairs `(return,policy)`,
- `evaluate_policies`: evaluates low level policies on several episodes,
- `evaluate_policies_high`: evaluates high level policies on several episodes.

Policies are modeled as objects. Their most important method is `drawAction`, but depending on the class some additional properties might be mandatory.

> **IMPORTANT!** All data is stored in **COLUMNS**, e.g., states are matrices `S x N`, where `S` is the size of one state and `N` is the number of states. Similarly, actions are matrices `A x N` and features are matrices `F x N`.
  </p>
  </details>


<details>
<summary><b>MDPs</b></summary>
  <p>
Each MDP is modeled as an object (<code>MDP.m</code>) and requires some properties (dimension of state and action spaces, bounds, etc...) and methods (for state transitions and plots). 
Each MDP also has a default discount factor <code>gamma</code>, which usually works well with the majority of the algorithms, but feel free to change it if necessary.  
The most important function is <code>[s',r,d] = simulator(s,a)</code>, which defines the transition function. 
The function returns <code>d = True</code> if the next state <code>s'</code> is terminal (episode ended). 
Usually, the reward <code>r(s,a,s')</code> depends on <code>s,a</code>, and on <code>s'</code> if the next state is terminal. 
For example, the cart-pole swing-up returns a reward depending on the current position of the pole + a penalty if the cart hits the walls (terminal next state).  

For MDPs sharing the same environment (e.g., mountain car with continuous or discrete actions, cart-pole with or without swing-up, ...), there are common <i>Environment</i> (<code>Env</code>) classes.
These classes define common variables and functions (transition, plotting, ...), while each subclass defines task-specific functions (reward, action parsing, terminal conditions, ...).

Finally, there are also subclasses for some special extension to MDPs, i.e., <i>Contextual MDPs</i> (<code>CMDP.m</code>), <i>Multi-objective MDPs</i> (<code>MOMDP.m</code>), and <i>Average-reward MDPs</i> (<code>MDP_avg.m</code>).  

> **IMPORTANT!** To allow parallel execution of multiple episodes, all MDPs functions (except the ones for plotting) need to support vectorized operations, i.e., they need to deal with states and actions represented as `S x N` and `A x N` matrices, respectively.
  </p>
</details>


<details>
<summary><b>MO_Library</b></summary>
  <p>
This folder contains functions used in the multi-objective framework, e.g., hypervolume estimators and Pareto-optimality filters.

> **IMPORTANT!** All frontiers are stored in **ROWS**, i.e., they are matrices `N x R`, where `N` is the number of points and `R` is the number of objectives.
  </p>
</details>


<details>
<summary><b>Utilities</b></summary>
  <p>
Utility functions used for matrix operations, plotting and sampling are stored in this folder.
  </p>
</details>


## How to make plots and render environments


Here is a list with examples of all ways for visualizing your data or rendering an environment. Please note that not all MDPs support rendering.

<details>
<summary><b>Real-time data plotting</b></summary>
  <p>
During the learning, it is possible to plot in real-time a desired variable (e.g., the expected return <code>J</code>) by using <code>updateplot</code>. 

```
updateplot('Return',iter,J,1)
```
  </p>
</details>  


<details>
<summary><b>Confidence interval plots from multiple trials</b></summary>
  <p>
If you are interested on evaluating an algorithm on several trials you can use the function <code>shadedErrorBar</code>. For a complete example, please refer to <code>make_stdplot.m</code>.
  </p>
</details>  


<details>
<summary><b>Real-time rendering</b></summary>
  <p>
Launch <code>mdp.showplot</code> to initialize the rendering of the agent-environment interaction will be shown during the learning. To stop plotting use <code>mdp.closeplot</code>.

> **IMPORTANT!** This is possible only if you are learning using one episode per iteration.
  </p>
</details>  


<details>
<summary><b>Offline rendering</b></summary>
  <p>

- For step-based algorithms, you can directly use the built-in plotting function of the MDPs.
As `collect_samples` returns a low-level dataset of the episodes, you just have to call `mdp.plotepisode`

```
data = collect_samples(mdp,episodes,steps,policy)
mdp.plotepisode(data(1),0.001)
```

- For episode-based algorithms, the low-level dataset is not returned. In this case, you can call `show_simulation`, which executes only one episode and renders it. This visualization can be used also in step-based algorithms.

```
show_simulation(mdp,policy,100,0.001)
show_simulation(mdp,policy.update(policy_high.drawAction(1)),100,0.001)
```

If the MDP provides pixels rendering, you can enable it by adding an additional argument to the function call

```
show_simulation(mdp,policy,100,0.001,1)
```
  </p>
</details>  


<details>
<summary><b>Plot policies</b></summary>
  <p>
If the state space is 2-dimensional, you can plot the value functions learned by policies and the action distribution over the states.

```
SettingDeep % deep sea treasure setup
RUN_PG % run policy gradient (terminate by CTRL+C)
policy.plotQ(mdp.stateLB,mdp.stateUB) % plot Q-function
policy.plotV(mdp.stateLB,mdp.stateUB) % plot V-function
policy.plotGreedy(mdp.stateLB,mdp.stateUB) % plot the action taken by zeroing the exploration
```
</p>
</details>  


<details>
<summary><b>MOMDPs Pareto frontier</b></summary>
  <p>
To plot a set of points as a Pareto frontier of a MOMDP, use <code>MOMDP.plotfront</code>. You can use additional arguments like in the built-in <code>plot</code> to customize the plot. Please note that the points have to be passed as rows and that the function does not filter dominated points.

```
MOMDP.plotfront([0.5 0.5; 1 0; 0 1], '--or', 'LineWidth', 2)
```
  </p>
</details>