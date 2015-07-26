function [ n_obj, policy, episodes, steps, gamma ] = resource_settings
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = resource_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

policy = gibbs(@resource_basis_pol, ...
    zeros(resource_basis_pol,1), ...
    mdp_vars.action_list);

episodes = 150;
steps = 40;

end

