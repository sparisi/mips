clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = PuddleworldContinuous;
robj = 1;

bfs = @(varargin)basis_krbf(7,[mdp.stateLB mdp.stateUB],0,varargin{:});
% bfs = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});

tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 4*eye(mdp.daction), size(x,2));
ds = collect_samples(mdp, 100, 100, tmp_policy);
B = avg_pairwise_dist([ds.s]);
bfs = @(varargin) basis_fourier(100, mdp.dstate, B, 0, varargin{:});

A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 4*eye(mdp.daction);
policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);
% policy = GaussianLinearChol(bfs, daction, A0, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = numel(A0);
mu0 = policy.theta(1:n_params);
Sigma0high = 10 * eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0)).^2;
Sigma0high = nearestSPD(Sigma0high);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 10000;
steps_eval = 100;
episodes_learn = 100;
steps_learn = 50;
