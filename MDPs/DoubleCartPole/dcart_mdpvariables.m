function mdp_vars = dcart_mdpvariables()
% This problem was originally presented in 
% A P Wieland, Evolving Controls for Unstable Systems (1991)
%
% However, this is a modified version, as presented in
% S Loscalzo et al., Predictive feature selection for genetic policy search 
% (2014)
%
% (It has a 'no-force' action, no pole friction, lower 'dt', different 
% constraints, different reward and higher mass/length of the shorter pole)

mdp_vars.nvar_state = 6;
mdp_vars.nvar_action = 1;
mdp_vars.nvar_reward = 1;
mdp_vars.action_list = [1,2,3];
mdp_vars.gamma = 1;
mdp_vars.isAvg = 0;
mdp_vars.isStochastic = 0;
mdp_vars.maxr = 1;

return
