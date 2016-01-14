clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
target = struct();

% Reach a pre-defined joint state
target.q   = @(t)[3/2*pi; 0];
target.qd  = @(t)[0; 0];
target.qdd = @(t)[0; 0];

% % Follow a pre-defined trajectory
% f1 = 2; f2 = 0.5;
% target.q   = @(t) wrapin2pi([ sin(2*pi*f1*t)-pi; sin(2*pi*f2*t) ]);
% target.qd  = @(t) [ 2*pi*f1*cos(2*pi*f1*t); 2*pi*f2*cos(2*pi*f2*t) ];
% target.qdd = @(t) [ -(2*pi*f1)^2*sin(2*pi*f1*t); -(2*pi*f2)^2*sin(2*pi*f2*t) ];

mdp = DoubleLink(target);
robj = 1;
dreward = mdp.dreward;
gamma = mdp.gamma;
daction = mdp.daction;

bfs = @(varargin)basis_poly(2, 4, 0, varargin{:});

A0 = zeros(daction,bfs()+1);
Sigma0 = 50*eye(daction);
policy = GaussianLinearDiag(bfs, daction, A0, Sigma0);
% policy = GaussianLinearChol(bfs, daction, A0, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
makeDet = 0; % 1 to learn deterministic low level policies
n_params = policy.dparams*~makeDet + numel(A0)*makeDet;
mu0 = policy.theta(1:n_params);
Sigma0high = 100 * eye(n_params);
Sigma0high = Sigma0high + diag(abs(mu0));
Sigma0high = nearestSPD(Sigma0high);
policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
% policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 100;
steps_eval = 2000;
episodes_learn = 100;
steps_learn = 1000;
