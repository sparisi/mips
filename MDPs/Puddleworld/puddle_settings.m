function [ n_obj, policy, episodes, steps, gamma ] = puddle_settings

mdp_vars = puddle_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

policy = gibbs(@puddle_basis_tile, ...
    zeros(puddle_basis_tile,1), ...
    mdp_vars.action_list);

episodes = 2000;
steps = 50;

end
