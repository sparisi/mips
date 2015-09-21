Description
-----------

MiPS (Minimal Policy Search) is a toolbox for Matlab providing the implementation of some of the most famous policy search algorithms, as well as some recent multi-objective methods and benchmark problems in reinforcement learning.

The only package required is the Statistics Toolbox.

Some utility functions are imported from MathWorks (original authors are always acknowledged).


Code Structure
--------------

Launch *INSTALL* to add the path of all folders:

- *Algs* contains all the algorithms,
- *Library* contains all the policies, solvers, generic basis functions, and functions for sampling and evaluation,
- *MDPs* contains the models and the simulators of the MPDs,
- *MO_Library* contains functions used in the multi-objective framework,
- *Utilities* contains utility functions.


How To Add A New MDP
--------------------

Each MDP requires some mandatory functions:

- *NAME_simulator*    : defines the reward and transition functions,

- *NAME_settings*     : defines the learning setup and returns
  - *n_obj*    : number of objectives of the problem,
  - *policy*   : policy used to solve the problem,
  - *episodes* : number of episodes used for evaluation / learning,
  - *steps*    : max number of steps of each episode,
  - *gamma*    : discount factor,

- *NAME_mdpvariables* : defines the details of the problem, i.e.,
  - *mdp_vars.nvar_state*   : dimensionality of the state,
  - *mdp_vars.nvar_action*  : dimensionality of the action,
  - *mdp_vars.nvar_reward*  : dimensionality of the reward,
  - *mdp_vars.maxr*         : maximum magnitude of the reward (used as normalization factor),
  - *mdp_vars.gamma*        : discount factor,
  - *mdp_vars.isAvg*        : 1 if we want to consider the average reward, 0 otherwise,
  - *mdp_vars.isStochastic* : 1 if the environment is stochastic, 0 otherwise.

Please notice that you have to manually change the number of episodes and steps in *NAME_settings* according to your needs (e.g., according to the algorithm used).

Additionally, there are some functions used to better organize the code:

- *NAME_environment*  : defines additional details of the problem environment,
- *NAME_basis*        : defines the features used to represent a state,
- *NAME_moref*        : returns all the details related to the multi-objective setup, i.e., the reference frontier used for comparison, the utopia and antiutopia points. It also returns a set of weights if the reference front is obtained with a weighted scalarization of the objectives. Both the frontier and the weights are saved in *NAME_ref.dat* and *NAME_w.dat*, respectively.

Finally, the function *settings_episodic* is used as a wrapper to set up the learning for episodic algorithms. Modify this function only to specify the distribution used to collect samples (e.g., a Gaussian with diagonal covariance or a Gaussian Mixture Model).


How the Simulator Works
-----------------------

Here is a short description of the main functions responsible for simulating the MDPs and collecting the relevant data.

- *execute*           : it is the lowest level function. It calls the specific simulators and runs a single episode,
- *collect_samples*   : it calls *execute* and returns a dataset with all the information about the simulated episodes (steps, action, reward, features),
- *collect_episodes*  : used for episodic RL. It calls *collect_samples* multiple times and returns only the relevant high-level information for episodic algorithms (parameters drawn at the beginning of the episode and cumulative reward at the end of it),
- *evaluate_policies* : a wrapper for calling *collect_samples* with an additional option to evaluate only deterministic policies. If the environment is also deterministic, the evaluation is done on a single episode,
- *evaluate_policies_ep* : similar to the previous function but for episodic algorithms.

Finally, for contextual policy search the same functions have *_ctx* appended at the end of their name.


ReLe Interface
--------------

For collecting samples and computing gradients and hessians, you can also use *ReLe*, a powerful toolbox in C. 
You can find it here: https://github.com/AIRLab-POLIMI/ReLe

First, you need to mex the files in */ReLe/rele_matlab/src/mexinterface* (you can use the wrapper *MEXMakefile*).
Then add such folder to the Matlab search path.
Finally just call *collect_samples_rele* instead of *collect_samples*.

Please notice that, unlike *collect_samples*, *collect_samples_rele* does not return the average entropy over the rollouts.
