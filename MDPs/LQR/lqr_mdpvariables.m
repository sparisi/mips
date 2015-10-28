function mdp_vars = lqr_mdpvariables()

dim = 5;
mdp_vars.dim = dim;
mdp_vars.nvar_state = dim;
mdp_vars.nvar_action = dim;
mdp_vars.nvar_reward = dim;
mdp_vars.gamma = 0.9;
mdp_vars.isAvg = 0;
mdp_vars.isStochastic = 0;
mdp_vars.maxr = ones(dim,1);

return
