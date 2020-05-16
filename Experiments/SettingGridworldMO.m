clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
N = 8;
mdp = GridworldMO(N);
robj = 1;

bfs = @(varargin)basis_poly(3,2,0,varargin{:});
% bfs = @(varargin)basis_krbf(N, [mdp.stateLB, mdp.stateUB], 0, varargin{:});

policy = Gibbs(bfs, zeros((bfs()+1)*(mdp.actionUB-1),1), mdp.actionLB:mdp.actionUB);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = policy.dparams;
mu0 = policy.theta(1:n_params);
Sigma0high = 1000 * eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0)).^2;
Sigma0high = nearestSPD(Sigma0high);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);
% policy_high = GmmConstant(mu0,Sigma0high,5);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 500;
episodes_learn = 250;
steps_eval = N*2;
steps_learn = N*2;
