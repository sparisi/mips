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
gamma = min(gamma,0.999999);
allactions = mdp.allactions;
nactions = length(allactions);

bfs = @(varargin)basis_krbf(7, [mdp.stateLB, mdp.stateUB], 1, varargin{:});
% bfs = @(varargin)basis_poly(2, mdp.dstate, 1, varargin{:});

d = bfs();
Qfun = @(s,theta)reshape(theta,d,nactions)'*bfs(s);
theta = zeros(d*nactions,1);

maxsteps = 100;
epsilon = 1;
policy_type = 'softmax';
lrate = 0.01;
iter = 1;
data = [];


%% Plot
npoints_plot = 10;
Xnodes = linspace(mdp.stateLB(1),mdp.stateUB(1),npoints_plot);
Ynodes = linspace(mdp.stateLB(2),mdp.stateUB(2),npoints_plot);
[X, Y] = meshgrid(Xnodes,Ynodes);
XY = [X(:)';Y(:)'];


%% Learn
while iter < 10000000
    
    state = mdp.initstate(1);
    endsim = 0;
    step = 0;
    
    while (step < maxsteps) && ~endsim
        step = step + 1;
    
        % Execute action and compute Q
        epsilon = max(0.1, epsilon * 0.9995);
        action = feval(policy_type,Qfun(state,theta),epsilon);
        [nextstate, reward, endsim] = mdp.simulator(state, action);
        phi_nexts = bfs(nextstate);
        Qn = Qfun(nextstate,theta);
        Q = Qfun(state,theta);

        % Compute target and error
        T = reward(robj) + gamma * max(Qn,[],1);
        E = T - Q(action);
    
        % Gradient update
        phi = zeros(size(theta));
        idx = (action-1)*d+1;
        phi(idx:idx+d-1) = bfs(state);
        theta = theta + lrate * E * phi;

        state = nextstate;
        iter = iter + 1;
    end
    
    Q = Qfun(XY,theta);
    V = max(Q,[],1);
    subimagesc('Q-function',Xnodes,Ynodes,Q)
    subimagesc('V-function',Xnodes,Ynodes,V)
    
end


%% Show
policy.drawAction = @(s)egreedy( Qfun(s,theta), 0 );
show_simulation(mdp, policy, .01, 100)
