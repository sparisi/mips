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
- *NAME_settings*     : defines the learning setup, i.e., the policy and the number of episodes and steps used for evaluation / learning.
- *NAME_mdpvariables* : defines the details of the problem, i.e.,
  - *mdp_vars.nvar_state*   : dimensionality of the state,
  - *mdp_vars.nvar_action*  : dimensionality of the action,
  - *mdp_vars.nvar_reward*  : dimensionality of the reward,
  - *mdp_vars.max_obj*      : reward normalization factor,
  - *mdp_vars.gamma*        : discount factor,
  - *mdp_vars.isAvg*        : 1 if we want to consider the average reward, 0 otherwise,
  - *mdp_vars.isStochastic* : 1 if the environment is stochastic, 0 otherwise.

Please notice that you have to manually change the number of episodes and steps in *NAME_settings* according to your needs (e.g., according to the algorithm used).

Additionally, there are some functions used to better organize the code:

- *NAME_environment*  : defines additional details of the problem environment,
- *NAME_basis*        : defines the features used to represent a state,
- *NAME_moref*        : returns all the details related to the multi-objective setup, i.e., the reference frontier used for comparison, the utopia and antiutopia points. It also returns a set of weights if the reference front is obtained with a weighted scalarization of the objectives. Both the frontier and the weights are saved in *NAME_ref.dat* and *NAME_w.dat*, respectively.

Finally, the function *settings_episodic* is used as a wrapper to set up the learning for episodic algorithms. Modify this function only to specify the distribution used to collect samples (e.g., a Gaussian with diagonal covariance or a Gaussian Mixture Model).


ReLe Interface
--------------

For collecting samples and computing gradients and hessians, you can also use *ReLe*, a powerful toolbox in C. 
You can find it here: https://github.com/AIRLab-POLIMI/ReLe

First, you need to mex the files in */ReLe/rele_matlab/src/MEX_functions* (you can use the wrapper *MEXMakefile*).
Then add such folder to the Matlab search path.
Finally just call *collect_samples_rele* instead of *collect_samples*.

Please notice that, unlike *collect_samples*, *collect_samples_rele* does not return the average entropy over the rollouts.
