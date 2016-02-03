clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = BicycleDriveContinuous;
robj = 1;
dreward = mdp.dreward;
gamma = mdp.gamma;
daction = mdp.daction;

bfs = @bicycledrive_basis_poly;

A0 = zeros(daction,bfs()+1);
Sigma0 = [2 0; 0 0.02];
policy = GaussianLinearDiag(bfs, daction, A0, Sigma0);
% policy = GaussianLinearChol(bfs, daction, A0, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
makeDet = 0; % 1 to learn deterministic low level policies
n_params = policy.dparams;
mu0 = policy.theta(1:n_params);
Sigma0high = eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0));
Sigma0high = nearestSPD(Sigma0high);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);
% policy_high = GmmConstant(mu0,Sigma0high,5);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 500;
steps_eval = 40000;
episodes_learn = 50;
steps_learn = 40000; % 36000 are the minimum number of timesteps to run 1 km (1km / vel / dt)
