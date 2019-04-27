clear all
close all

rng(2)

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = CollectRewards2;
robj = 1;

tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 1*eye(mdp.daction), size(x,2));
ds = collect_samples(mdp, 200, 50, tmp_policy);
s = [ds.s];
B = avg_pairwise_dist(s) + 1e-6;
bfs = @(varargin) basis_fourier(50, mdp.dstate, B, 0, varargin{:});
bfs = @(varargin)basis_krbf(10, [[-20;-20],[20;20]], 0, varargin{:});

A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 0.5 * eye(mdp.daction);
policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);
% policy = GaussianLinearChol(bfs, mdp.daction, A0, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = numel(A0);
mu0 = policy.theta(1:n_params);
Sigma0high = 100 * eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0)).^2;
Sigma0high = nearestSPD(Sigma0high);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 1;
steps_eval = 100;
episodes_learn = 20;
steps_learn = 45;
