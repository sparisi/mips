function [ n_obj, policy, episodes, steps, gamma ] = mcs_settings

mdp_vars = mcs_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

policy = gibbs(@mcs_basis_poly, ...
    zeros(mcs_basis_poly,1), ...
    mdp_vars.action_list);

episodes = 100;
steps = 10000;

end
