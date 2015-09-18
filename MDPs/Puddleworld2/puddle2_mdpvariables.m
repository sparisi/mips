function mdp_vars = puddle2_mdpvariables()
% Reference: Vamplew et al, Empirical evaluation methods for multiobjective 
% reinforcement learning algorithms (2011)

mdp_vars.nvar_state = 2;
mdp_vars.nvar_action = 2;
mdp_vars.nvar_reward = 1;
mdp_vars.gamma = 1;
mdp_vars.isAvg = 0;
mdp_vars.isStochastic = 1; % random init position
mdp_vars.maxr = 1;

return