function [ n_obj, policy, episodes, steps, gamma ] = deep_settings
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = deep_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

policy = gibbs(@deep_basis_pol_v1, ...
    zeros(deep_basis_pol_v1,1), ...
    mdp_vars.action_list);

%%% Evaluation
episodes = 500;
steps = 50;

%%% Learning
episodes = 150;
steps = 50;

end
