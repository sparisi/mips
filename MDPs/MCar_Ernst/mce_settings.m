function [ n_obj, policy, episodes, steps, gamma ] = mce_settings
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = mce_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

policy = gibbs(@mce_basis_rbf_v1, ...
    zeros(mce_basis_rbf_v1,1), ...
    mdp_vars.action_list);

episodes = 500;
steps = 150;

end
