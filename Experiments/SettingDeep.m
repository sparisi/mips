clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = DeepSeaTreasure;
robj = 1;
dreward = mdp.dreward;
gamma = mdp.gamma;
nactions = mdp.actionUB;

bfs = @deep_basis_poly;
bfs = @(varargin)basis_krbf(11, [mdp.stateLB, mdp.stateUB], 0, varargin{:});

% policy = Gibbs(bfs, zeros((bfs()+1)*(nactions-1),1), mdp.actionLB:mdp.actionUB);
policy = EGreedy(bfs, zeros((bfs()+1)*nactions,1), mdp.actionLB:mdp.actionUB, 1);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
makeDet = 1; % 1 to learn deterministic low level policies
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
if makeDet
    episodes_eval = 1;
    episode_learn = 1;
else
    episodes_eval = 500;
    episodes_learn = 250;
end
steps_eval = 50;
steps_learn = 50;
