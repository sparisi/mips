function [ n_obj, policy, episodes, steps, gamma ] = deep_settings

mdp_vars = deep_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

policy = gibbs(@deep_basis_poly, ...
    zeros(deep_basis_poly,1), ...
    mdp_vars.action_list);

%%% Evaluation
episodes = 500;
steps = 50;

%%% Learning
episodes = 250;
steps = 50;

end
