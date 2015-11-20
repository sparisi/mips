function mdp_vars = mcc_mdpvariables()
% Reference: Ernst et al, Tree-Based Batch Mode Reinforcement Learning
% (2005)

mdp_vars.nvar_state = 2;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 1;
mdp_vars.gamma = 1;
mdp_vars.isAvg = 0;
mdp_vars.isStochastic = 0;
mdp_vars.maxr = 1;

return
