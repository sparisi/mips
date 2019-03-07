% Natural Actor-Critic with LSTD-Q(lambda).
%
% =========================================================================
% REFERENCE
% J Peters, S Vijayakumar, S Schaal
% Natural Actor-Critic (2005)


% bfsV = @(varargin)basis_krbf(7, [mdp.stateLB, mdp.stateUB], 1, varargin{:});
% bfsV = @(varargin)basis_poly(2, mdp.dstate, 1, varargin{:});
% tmp_policy.drawAction = @(x)mymvnrnd(zeros(mdp.daction,1), 16*eye(mdp.daction), size(x,2));
% ds = collect_samples(mdp, 100, 100, tmp_policy);
% BW = avg_pairwise_dist([ds.s]);
% bfsV = @(varargin) basis_fourier(50, mdp.dstate, BW, 0, varargin{:});

bfsV = bfs;

w = zeros(bfsV(),1);

A = eye(bfsV() + policy.dparams);
b = 0;
z = 0;
beta = 0.8; % Works for PuddleworldContinuous
beta = 0.99; % Works for Pendulum

lambda = 0.95;
gamma = 0.99;
lrate = 0.01;
l2_reg = 0.001;

eval_every = 200; % evaluation frequency (in steps)
update_every = 1; % update frequency (in steps)
minsteps = 100; % wait to collect some samples before updating

totsteps = 0;
J_history = [];


%% Plot
if mdp.dstate == 2 && ~sum(isinf(mdp.stateLB) | isinf(mdp.stateUB))
npoints_plot = 10;
Xnodes = linspace(mdp.stateLB(1),mdp.stateUB(1),npoints_plot);
Ynodes = linspace(mdp.stateLB(2),mdp.stateUB(2),npoints_plot);
[X, Y] = meshgrid(Xnodes,Ynodes);
XY = [X(:)';Y(:)'];
end

%% Learn
for episode = 1 : 100000
    
    step = 0;
    state = mdp.initstate(1);
    terminal = 0;
    
    % Run the episodes until maxsteps or terminal state
    while (step < steps_learn) && ~terminal
        
        step = step + 1;
        action = policy.drawAction(state);
        
        % Simulate one step
        [nextstate, reward, terminal] = mdp.simulator(state, action);

        % LSTD-Q(lambda)
        phi_tilde = [bfsV(nextstate); zeros(policy.dparams,1)];
        phi_hat = [bfsV(state); policy.dlogPidtheta(state,action)];
        
        z = lambda * z + phi_hat;
        A = A + z * (phi_hat - gamma*phi_tilde)';
        b = b + z * reward;
        
        if rank(A) == size(A,1)
            wv = A \ b;
        else
%             wv = pinv(A) * b;
            wv = (A + l2_reg*eye(size(A))) \ b;
        end
        
        v = wv(1:bfsV());
        w = wv(bfsV()+1:end);
        
        % Forget statistics
        A = beta * A;
        b = beta * b;
        z = beta * z;

        % Policy gradient
        if totsteps > minsteps && mod(totsteps, update_every) == 0
            policy = policy.update(policy.theta + lrate * w / max(1,norm(w)));
        end
        
        % Continue
        state = nextstate;
        totsteps = totsteps + 1;

        % Evaluate
        if mod(totsteps, eval_every) == 0
            J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
            if mdp.dstate == 2 && ~sum(isinf(mdp.stateLB) | isinf(mdp.stateUB))
                updatesurf('V-function',X,Y,reshape(v'*bfsV(XY),[npoints_plot,npoints_plot]))
            end
            J_history(end+1) = J;
            fprintf('%d | %e  %e\n', totsteps, J, norm(w));
        end
        
    end
    
end


%% Show
show_simulation(mdp, policy.makeDeterministic, 100, 0.01)
