function mdp_vars = mcs_mdpvariables()
% Reference: Vamplew et al, Empirical evaluation methods for multiobjective 
% reinforcement learning algorithms (2011)

mdp_vars.nvar_state = 2;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 3;
mdp_vars.action_list = [1,2,3];
mdp_vars.gamma = 1;
mdp_vars.isAvg = 0;
mdp_vars.isStochastic = 0;
mdp_vars.maxr = [1;1;1];

return