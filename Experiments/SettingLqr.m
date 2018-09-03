clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
dim = 2;
% mdp = LQR_MO(dim);
mdp = LQR(dim);
% mdp = LQR_MM(dim,[-5*ones(dim,1), 5*ones(dim,1)]);

robj = 1;

bfs = @(varargin)basis_poly(1,dim,0,varargin{:});
% bfs = @(varargin)basis_krbf(10, 20*[-ones(dim,1), ones(dim,1)], 0, varargin{:});
% bfs = @(varargin)basis_rbf(5*[-ones(dim,1), ones(dim,1)], 0.5./[5; 5], 0, varargin{:});

A0 = zeros(dim,bfs()+1);
Sigma0 = 5*eye(dim);
% policy = GaussianLinearChol(bfs, dim, A0, Sigma0);
% policy = GaussianLinearDiag(bfs, dim, A0, Sigma0);
% policy = GaussianLinearFixedvar(bfs, dim, A0, Sigma0);
% policy = GaussianLinearFixedvarDiagmean(bfs, dim, A0(:,2:end), Sigma0);
policy = GaussianLinearFixedvarDiagmean(bfs, dim, -diag(rand(dim,1))*0.1, Sigma0);


%% ===================================================================== %%
%  ======================= HIGH LEVEL SETTINGS =========================  %
n_params = numel(A0);
if isa(policy,'GaussianLinearFixedvarDiagmean'), n_params = dim; end
mu0 = policy.theta(1:n_params);
Sigma0high = 1 * eye(n_params);
tau = diag(Sigma0high);
% policy_high = GaussianConstantChol(n_params, mu0, Sigma0high);
policy_high = GaussianConstantDiag(n_params, mu0, Sigma0high);


%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 150;
steps_eval = 150;
episodes_learn = 10;
steps_learn = 150;
