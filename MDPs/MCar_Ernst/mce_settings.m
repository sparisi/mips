function [ n_obj, policy, episodes, steps, gamma ] = mce_settings

mdp_vars = mce_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

policy = gibbs(@mce_basis_rbf, ...
    zeros(mce_basis_rbf,1), ...
    mdp_vars.action_list);

episodes = 500;
steps = 150;

end
