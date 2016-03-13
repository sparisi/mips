clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = BicycleDrive;
robj = 1;
dreward = mdp.dreward;
gamma = mdp.gamma;
nactions = mdp.actionUB;

bfs = @bicycledrive_basis_poly;

% policy = EGreedy(bfs, zeros((bfs()+1)*(nactions-1),1), mdp.actionLB:mdp.actionUB, 1);
policy = Gibbs(bfs, zeros((bfs()+1)*(nactions-1),1), mdp.actionLB:mdp.actionUB);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
makeDet = 0; % 1 to learn deterministic low level policies
n_params = policy.dparams;
mu0 = policy.theta(1:n_params);
Sigma0high = 10 * eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0)).^2;
Sigma0high = nearestSPD(Sigma0high);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);
% policy_high = GmmConstant(mu0,Sigma0high,5);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 1000;
steps_eval = 40000;
episodes_learn = 250;
steps_learn = 40000; % 36000 are the minimum number of timesteps to run 1 km (1km / vel / dt)
