% Least-Squares Policy Iteration.
%
% =========================================================================
% REFERENCE
% M G Lagoudakis, R Parr, M L Littman
% Least-Squares Methods in Reinforcement Learning for Control (2002)

clear all
close all

mdp = MCar;
% mdp = DeepSeaTreasure;
% mdp = Gridworld;
% mdp = Puddleworld;
% mdp = CartPole;
% mdp = BicycleDrive;

robj = 1;
gamma = mdp.gamma;
allactions = 1 : size(mdp.allactions,2);
nactions = length(allactions);

bfs = @(varargin)basis_krbf(7, [mdp.stateLB, mdp.stateUB], 1, varargin{:});
% bfs = @(varargin)basis_poly(2, mdp.dstate, 1, varargin{:});

d = bfs();
Qfun = @(s,theta)reshape(theta,d,nactions)'*bfs(s);
theta = zeros(d*nactions,1);

episodes = 100;
maxsteps = 100;
epsilon = 1;
policy_type = 'softmax';
A = 0;
b = 0;
lambda = 1;
expl_rate = 1; % If <1, it decreases the exploration while learning
iter = 1;


%% Plot
npoints_plot = 10;
Xnodes = linspace(mdp.stateLB(1),mdp.stateUB(1),npoints_plot);
Ynodes = linspace(mdp.stateLB(2),mdp.stateUB(2),npoints_plot);
[X, Y] = meshgrid(Xnodes,Ynodes);
XY = [X(:)';Y(:)'];


%% Learn
while iter < 100000

    % Set policy
    epsilon = max(0.1, epsilon * expl_rate);
    policy.drawAction = @(s)feval(policy_type,Qfun(s,theta),epsilon);

    % Collect data and compute Q
    data = collect_samples2(mdp, episodes, maxsteps, policy);
    Qn = Qfun(data.nexts,theta);
    Q = Qfun(data.s,theta);
    a_nexts = feval(policy_type,Qn,0);

    % Compute target and error
    T = data.r(robj,:) + gamma * max(Qn,[],1);
    E = Q((0:length(data.a)-1)*nactions+data.a) - T;
    
    % Compute features
    phi = shiftvec(bfs(data.s), data.a, max(allactions));
    phi_nexts = shiftvec(bfs(data.nexts), a_nexts, max(allactions));

    % L2 linear regression
    A = A + phi * (phi - gamma * phi_nexts)';
    b = b + phi * data.r(robj,:)';
    theta_old = theta;
    theta = (A + lambda * eye(size(A))) \ b;
    diff = norm(theta-theta_old);
    
    % Plot
    updateplot('Error',iter,mean(E.^2),1)
    updateplot('Diff',iter,diff,1)
    Q = Qfun(XY,theta);
    V = max(Q,[],1);
    subimagesc('Q-function',Xnodes,Ynodes,Q)
    subimagesc('V-function',Xnodes,Ynodes,V)
    if iter == 1, autolayout, end

    iter = iter + 1;

end


%% Show
policy_eval.drawAction = @(s)egreedy( Qfun(s,theta), 0 );
show_simulation(mdp, policy_eval, .01, 100)
