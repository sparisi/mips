function [ n_obj, policy, episodes, steps, gamma ] = dam_settings
% Outputs:
%
% - n_obj    : number of objectives of the MDP
% - policy   : default policy
% - episodes : number of episodes for evaluation / learning
% - steps    : max number of steps per episode
% - gamma    : discount factor of the MDP

mdp_vars = dam_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;
dim = mdp_vars.nvar_action;

k0 = [50, -50, 0, 0, 50];
% k0 = zeros(1,dam_basis_rbf());
% policy = gaussian_fixedvar(@dam_basis_rbf, dim, k0, 0.1);
% policy = gaussian_linear(@dam_basis_rbf, dim, k0, 20);
policy = gaussian_diag_linear(@dam_basis_rbf, dim, k0, 20);
% policy = gaussian_logistic_linear(@dam_basis_rbf, dim, k0, 1, 50);

%%% Evaluation
episodes = 1000;
steps = 100;

%%% Learning
episodes = 50;
steps = 30;

end

