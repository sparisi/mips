function [ n_obj, policy, episodes, steps, gamma ] = mountain_settings
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = mountain_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

policy = gibbs(@mountain_basis_pol, ...
    zeros(mountain_basis_pol,1), ...
    mdp_vars.action_list);

episodes = 100;
steps = 10000;

end
