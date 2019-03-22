clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = Puddleworld;
robj = 1;

bfs = @(varargin)basis_krbf(7,[mdp.stateLB mdp.stateUB],0,varargin{:});
% bfs = @(varargin)basis_poly(2,mdp.dstate,0,varargin{:});

% tmp_policy.drawAction = @(x) randi(mdp.actionUB, 1, size(x,2));
% ds = collect_samples(mdp, 100, 100, tmp_policy);
% B = avg_pairwise_dist([ds.s]);
% bfs = @(varargin) basis_fourier(50, mdp.dstate, B, 0, varargin{:});

policy = Gibbs(bfs, zeros((bfs()+1)*(mdp.actionUB-1),1), mdp.actionLB:mdp.actionUB);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = policy.dparams;
mu0 = policy.theta(1:n_params);
Sigma0high = 10 * eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0)).^2;
Sigma0high = nearestSPD(Sigma0high);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 500;
steps_eval = 100;
episodes_learn = 250;
steps_learn = 50;
