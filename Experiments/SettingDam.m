clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = Dam(2);
mdp.penalize = 0;
robj = 1;

bfs = @dam_basis_rbf;

A0 = [50, -50, 0, 0, 50];
% A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 50^2;
% policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);
policy = GaussianLinearChol(bfs, mdp.daction, A0, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = numel(A0);
mu0 = policy.theta(1:n_params);
Sigma0high = 2500 * eye(n_params);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 1000;
steps_eval = 100;
episodes_learn = 100;
steps_learn = 100;
