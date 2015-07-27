function mdp_vars = mce_mdpvariables()
% Reference: Ernst et al, Tree-Based Batch Mode Reinforcement Learning

mdp_vars.nvar_state = 2;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 1;
mdp_vars.action_list = [1,2,3];
mdp_vars.max_obj = 1;
mdp_vars.gamma = 0.9;
mdp_vars.isAvg = 0;
mdp_vars.isStochastic = 0;

return
