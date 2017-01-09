% Compatible Off-Policy Deterministic Actor-Critic.
% You can use a Q-learning critic or a Gradient Q-learning critic.
% Both the policy and the Q-function approximation are linear in the
% parameters.
% 
% =========================================================================
% REFERENCE
% D Silver, G Lever, N Heess, T Degris, D Wierstra, M Riedmiller
% Deterministic Policy Gradient Algorithms (2014)

clear all
close all

rng(1)

showplots = 1;
J_history = [];

mdp = ChainwalkContinuous; sigma = 1; maxsteps = 20;
% mdp = ChainwalkContinuousMulti(2); sigma = 2; maxsteps = 20;
% mdp = MCarContinuous; sigma = 4; maxsteps = 100;
% mdp = CartPoleContinuous; sigma = 4; maxsteps = 1000;
mdp = PuddleworldContinuous; sigma = 0.2; maxsteps = 100;

robj = 1;
gamma = mdp.gamma;
noise = @() mvnrnd(zeros(mdp.daction,1),sigma*eye(mdp.daction))';
lrate_theta = 0.01;
lrate_v = 0.001;
lrate_w = 0.001;
iter = 1;


%% Setup actor and critic
% Policy
bfs_mu = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 1, varargin{:});
% bfs_mu = @(varargin)basis_poly(2, mdp.dstate, 1, varargin{:});
theta = zeros(bfs_mu()*mdp.daction,1);
mu = @(s,theta) kron(eye(mdp.daction),bfs_mu(s))' * theta;
mu = @(s,theta) reshape(theta,bfs_mu(),mdp.daction)' * bfs_mu(s);
d_policy = @(s) kron(eye(mdp.daction),bfs_mu(s));

% V-function
bfs_v = @(varargin)basis_krbf(4, [mdp.stateLB, mdp.stateUB], 1, varargin{:});
% bfs_v = @(varargin)basis_poly(2, mdp.dstate, 1, varargin{:});
v = zeros(bfs_v(),1);
Vfun = @(s,v) v'*bfs_v(s);

% Q-function (with compatible function approximation)
w = zeros(size(theta));
bfs_q = @(s,a,theta) d_policy(s) * (a - mu(s,theta));
Qfun = @(s,a,w,v,theta) bfs_q(s,a,theta)' * w + Vfun(s,v);

u = zeros(size(w));
lrate_u = lrate_v;

%% Prepare plots
if showplots
    if mdp.dstate == 2
        npoints_plot = 10;
        Xnodes = linspace(mdp.stateLB(1),mdp.stateUB(1),npoints_plot);
        Ynodes = linspace(mdp.stateLB(2),mdp.stateUB(2),npoints_plot);
        [X, Y] = meshgrid(Xnodes,Ynodes);
        XY = [X(:)';Y(:)'];
        S = XY;
    elseif mdp.dstate == 1
        npoints_plot = 10;
        S = linspace(mdp.stateLB(1),mdp.stateUB(1),npoints_plot);
        figure; hP = plot(S,zeros(1,npoints_plot)); title 'pi(s)';
        figure; hQ = plot(S,zeros(1,npoints_plot)); title 'Q(s,pi(s))';
    end
    mdp.showplot
end

%% Learn
while iter < 3000000
    
    state = mdp.initstate(1);
    endsim = 0;
    step = 0;
    
    while (step < maxsteps) && ~endsim
        step = step + 1;
    
        action = mu(state,theta) + noise();
        [nextstate, reward, endsim] = mdp.simulator(state, action);
        Qn = Qfun(nextstate, mu(nextstate,theta), w, v, theta); % Off-policy
        Q = Qfun(state, action, w, v, theta);

        % Actor and critic update
        delta = reward(robj) + gamma * Qn * ~endsim - Q;
%         theta = theta + lrate_theta * d_policy(state) * (d_policy(state)' * w); % Vanilla
        theta = theta + lrate_theta * w; % Natural

        % Q-learning critic
%         w = w + lrate_w * delta * bfs_q(state,action,theta);
%         v = v + lrate_v * delta * bfs_v(state);

        % Gradient Q-learning critic
        w = w + lrate_w * delta * bfs_q(state,action,theta) - ...
            lrate_w * gamma * bfs_q(nextstate,mu(nextstate,theta),theta) * (bfs_q(state,action,theta)' * u);
        v = v + lrate_v * delta * bfs_v(state) - ...
            lrate_v * gamma * bfs_v(nextstate) * (bfs_q(state,action,theta)' * u);
        u = u + lrate_u * (delta - (bfs_q(state,action,theta)' * u)) * bfs_q(state,action,theta);
        
        if any(isnan(theta)) || any(isnan(w)) || any(isnan(v)) || ...
                any(isinf(theta)) || any(isinf(w)) || any(isinf(v))
            error('Inf or NaN.')
        end
        
        state = nextstate;
        iter = iter + 1;

        % Plot
        if showplots
            if mdp.dstate == 1
                P = mu(S,theta);
                for i = 1 : size(S,2)
                    Q(:,i) = Qfun(S(:,i),P(:,i),w,v,theta);
                end
                hP.YData = P;
                hQ.YData = Q;
                drawnow limitrate
            elseif mdp.dstate == 2
                P = mu(S,theta);
                for i = 1 : size(S,2)
                    Q(:,i) = Qfun(S(:,i),P(:,i),w,v,theta);
                end
                subimagesc('pi(s)',Xnodes,Ynodes,P,1)
                subimagesc('Q(s,pi(s))',Xnodes,Ynodes,Q)
            end
            updateplot('delta',iter,delta^2)
            updateplot('action',iter,action)
            if iter == 2, autolayout, end
        end
        
    end
    
    policy_eval.drawAction = @(s) mu(s,theta);
    J_history(end+1) = evaluate_policies(mdp, 100, maxsteps, policy_eval);
    if showplots, updateplot('J',iter,J_history(end)); end

end


%% Show
policy_eval.drawAction = @(s) mu(s,theta);
[J, ds] = show_simulation(mdp, policy_eval, 50, 0.01);
J
