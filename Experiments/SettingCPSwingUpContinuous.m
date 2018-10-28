clear all
close all

rng(1)

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = CartPoleSwingUpContinuous;
robj = 1;

tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 10*eye(mdp.daction), size(x,2));
ds = collect_samples(mdp, 100, 100, tmp_policy);
B = avg_pairwise_dist([ds.s]);
bfs = @(varargin) basis_fourier(200, mdp.dstate, B, 0, varargin{:});

A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 10^2;
% policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);
policy = GaussianLinearChol(bfs, mdp.daction, A0, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = policy.dparams;
mu0 = policy.theta(1:n_params);
Sigma0high = 100 * eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0)).^2;
Sigma0high = nearestSPD(Sigma0high);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 500;
steps_eval = 1000;
episodes_learn = 500;
steps_learn = 100;
