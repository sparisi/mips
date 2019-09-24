clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = ChainwalkContBonus;
robj = 1;

% bfs = @(varargin) basis_poly(2, mdp.dstate, 0, varargin{:});
bfs = @(varargin) basis_krbf(10, [mdp.stateLB, mdp.stateUB], 0, varargin{:});

% tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 1*eye(mdp.daction), size(x,2));
% ds = collect_samples(mdp, 100, 100, tmp_policy);
% B = avg_pairwise_dist([ds.s]);
% bfs = @(varargin) basis_fourier(100, mdp.dstate, B, 0, varargin{:});

A0 = zeros(mdp.daction,bfs()+1);
Sigma0 = 1*eye(mdp.daction);
policy = GaussianLinearDiag(bfs, mdp.daction, A0, Sigma0);

%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 1000;
steps_eval = 500;
episodes_learn = 50;
steps_learn = 200;
