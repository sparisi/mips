clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = DoubleLink;
% mdp = QuadLink;
% mdp = Reacher;

robj = 1;

% bfs = @(varargin) basis_pixels(mdp,varargin{:});
bfs = @(varargin) basis_poly(2, mdp.dstate, 0, varargin{:});
bfs = @(varargin) basis_krbf(4, [mdp.stateLB, mdp.stateUB], 0, varargin{:});

tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 100*eye(mdp.daction), size(x,2));
ds = collect_samples(mdp, 400, 100, tmp_policy);
state = [ds.s];
B = avg_pairwise_dist([cos(state(1:2:end,:)); sin(state(1:2:end,:)); state(2:2:end,:)]);
bfs_base = @(varargin) basis_fourier(300, mdp.dstate+mdp.dstate/2, B, 0, varargin{:});
bfs = @(varargin)basis_nlink(bfs_base, varargin{:});

A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 200*eye(mdp.daction);
policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);
policy = GaussianLinearChol(bfs, mdp.daction, A0, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = numel(A0);
mu0 = policy.theta(1:n_params);
Sigma0high = 10 * eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0)).^2;
Sigma0high = nearestSPD(Sigma0high);
% policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 1000;
steps_eval = 500;
episodes_learn = 50;
steps_learn = 500;
