clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = BicycleDriveContinuous;
robj = 1;

bfs = @bicycledrive_basis_poly;
% bfs = @(varargin) basis_poly(2,mdp.dstate,0,varargin{:});

A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 20*[4 0; 0 0.04];
% policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);
policy = GaussianLinearChol(bfs, mdp.daction, A0, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = policy.dparams;
mu0 = policy.theta(1:n_params);
Sigma0high = 10*eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0)).^2;
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
