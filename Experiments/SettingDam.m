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
makeDet = 0; % 1 to learn deterministic low level policies
n_params = policy.dparams*~makeDet + numel(A0)*makeDet;
mu0 = policy.theta(1:n_params);
Sigma0high = 2500 * eye(n_params);
% Sigma0high = Sigma0high + diag(abs(mu0)).^2;
Sigma0high = nearestSPD(Sigma0high);
if makeDet == 0
mu0(end) = 0;
Sigma0high(end,end) = 50;
end
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 1000;
steps_eval = 100;
episodes_learn = 100;
steps_learn = 100;
