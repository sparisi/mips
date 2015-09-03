function mdp_vars = puddle_mdpvariables()
% Reference: Vamplew et al, Empirical evaluation methods for multiobjective 
% reinforcement learning algorithms (2011)

mdp_vars.nvar_state = 2;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 2;
mdp_vars.action_list = [1,2,3,4];
mdp_vars.gamma = 1;
mdp_vars.isAvg = 0;
mdp_vars.isStochastic = 1; % random init position
mdp_vars.maxr = [1;1];

return