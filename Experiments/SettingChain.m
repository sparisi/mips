clear all
close all

%% ===================================================================== %%
%  ======================== LOW LEVEL SETTINGS =========================  %
mdp = ChainwalkContinuous;
% mdp = ChainwalkContinuousMulti(2);
robj = 1;
dreward = mdp.dreward;
gamma = mdp.gamma;
daction = mdp.daction;

bfs = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 1, varargin{:});

A0 = zeros(daction,bfs()+1);
Sigma0 = 1*eye(daction);
policy = GaussianLinearDiag(bfs, daction, A0, Sigma0);
% policy = GaussianLinearChol(bfs, daction, A0, Sigma0);

%% ===================================================================== %%
%  ======================== LEARNING SETTINGS ==========================  %
episodes_eval = 500;
steps_eval = 100;
episodes_learn = 150;
steps_learn = 40;
