function mdp_vars = damC_mdpvariables()
% Reference: Castelletti et al, Tree-based fitted q-iteration for MOMDP
% (2013)

mdp_vars.nvar_state = 1;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 1;
mdp_vars.gamma = 1;
mdp_vars.isAvg = 1;
mdp_vars.isStochastic = 1; % random init state and random inflow
mdp_vars.maxr = 1;

return
