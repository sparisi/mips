classdef DDPG_Solver < handle
% DDPG Deep Deterministic Policy Gradent.
%
% =========================================================================
% REFERENCE
% T P Lillicrap, J J Hunt, A Pritzel, N Heess, T Erez, Y Tassa, D Silver, 
% D Wierstra
% Continuous control with deep reinforcement learning (2016)
% http://arxiv.org/abs/1509.02971
    
    properties

        %% Hyperparameters
        bsize = 32;         % Minibatch size
        dsize = 1e6;        % Database size
        gamma = 0.99;       % Discount factor
        maxsteps = 1e3;     % Max steps per episode
        tau = 1e-2          % Update coefficient for the target networks
        sigma = 2           % Noise on the action (std)
        noise_decay = 0.9999; % Decay of the exploration during an episode
        
        %% Functions
        mdp                  % MDP with functions for init an episode and performing a step
        preprocessS = @(s)s; % Preprocess the state
        preprocessR = @(r)r; % Preprocess the reward
        
        %% Networks
        nnP  % Policy
        nnPt % Policy target
        nnQ  % Q-function
        nnQt % Q-function target
        
        %% Optimizers
        optimP % Optimizer for the policy
        optimQ % Optimizer for the Q-function
        
        %% Initialization
        t = 0;  % Total elapsed timesteps
        data    % Replay buffer
        dimA    % Number of actions
        
    end
    
    methods

        %% Constructor
        function obj = DDPG_Solver(nnP, nnQ, optimP, optimQ, dimA, dimO, mdp)
            obj.dimA        = dimA;
            obj.optimP      = optimP;
            obj.optimQ      = optimQ;
            obj.data.o      = NaN(obj.dsize,dimO);
            obj.data.a      = NaN(obj.dsize,dimA);
            obj.data.r      = NaN(obj.dsize,1);
            obj.data.o_next = NaN(obj.dsize,dimO);
            obj.data.term   = NaN(obj.dsize,1);
            obj.mdp         = mdp;
            obj.nnP         = nnP;
            % If MDP action is bounded in [-A,A], add a tanh final layer to 
            % the policy network to bound its output
            if ~any(isinf(obj.mdp.actionUB))
                nnP.set_output('Tanh');
            end
            obj.nnQ         = nnQ;
            obj.nnPt        = copy(nnP);
            obj.nnQt        = copy(nnQ);
        end
        
        function [J, avgL, states] = train(obj)
            % Get init state and initialize variables
            state = obj.mdp.initstate(1);
            states = zeros(obj.maxsteps, length(state));
            J = 0; % Expected return
            L = []; % Loss on the Q-function
            step = 0;
            terminal = false;
            
            obj.sigma = obj.noise_decay*obj.sigma;
            
            while ~terminal && step < obj.maxsteps
                step = step + 1;
                states(step, :) = state;

                obs = obj.preprocessS(state);
                action = forward(obj.nnP,obs);
                noise = mymvnrnd(0, obj.sigma.^2, obj.dimA);
                action = action + noise;

                % If MDP action is bounded in [-A,A], the policy network 
                % has a tanh final layer to bound its output
                if any(isinf(obj.mdp.actionUB))
                    [nextstate, reward, terminal] = obj.mdp.simulator(state, action');
                else
                    [nextstate, reward, terminal] = obj.mdp.simulator(state, obj.mdp.actionUB.*action');
                end
                obs_next = obj.preprocessS(nextstate);
                reward = obj.preprocessR(reward);
                
                obj.t = obj.t + 1;
                idx = mod(obj.t-1,obj.dsize)+1;
                obj.data.o(idx,:) = obs;
                obj.data.a(idx,:) = action;
                obj.data.r(idx,:) = reward;
                obj.data.o_next(idx,:) = obs_next;
                obj.data.term(idx,:) = terminal;
                
                J = J + obj.gamma^(step-1)*reward;
                state = nextstate;
                
                if(obj.t > obj.bsize), L(end+1) = obj.step(); end
            end
            
            states = states(1:step,:);
            avgL = mean(L);
        end
        
        function L = step(obj)
            mb = randperm(min(obj.t,obj.dsize),obj.bsize); % Random minibatches
            
            O_next  = obj.data.o_next(mb,:); % bsize x dimO
            O       = obj.data.o(mb,:);      % bsize x dimO
            A       = obj.data.a(mb,:);      % bsize x dimA
            R       = obj.data.r(mb,:);      % bsize x 1
            Term    = obj.data.term(mb,:);   % bsize x 1
            
            % Compute targets via Bellman equation with target network
            A_next  = forward(obj.nnPt,O_next);
            QT_next = forward(obj.nnQt,[A_next O_next]);
            T       = R + obj.gamma .* QT_next .* ~Term;
            
            % Compute error, loss and gradients
            Q  = forwardfull(obj.nnQ,[A O]);
            E  = Q - T;
            L  = mean(E.^2);
            dL = 2 / obj.bsize * E;

            % Critic update (Q-networks)
            dW_q = backward(obj.nnQ,dL);
            obj.nnQ.update(step(obj.optimQ, obj.nnQ.W, dW_q));
            obj.nnQt.update(obj.tau * obj.nnQ.W + (1 - obj.tau) * obj.nnQt.W); % Soft update of the target network

            % Actor update (policy networks)
            A_det = forwardfull(obj.nnP,O); % Deterministic actions (no noise)
            Q_det = forwardfull(obj.nnQ,[A_det O]); % Q-function of deterministic actions
            [~, dL_q] = backward(obj.nnQ,ones(size(Q_det))); % Derivative wrt [A_det O] of the Q-network
            dL_q = dL_q(:,1:obj.dimA); % dL_q is the input of the Q-network and the first components are also the output of the policy-network
            dW_p = -backward(obj.nnP,dL_q); % The minus is to transform the problem from a minimization to a maximization (we are computing the derivative of the expected return J)
            obj.nnP.update(step(obj.optimP, obj.nnP.W, dW_p));
            obj.nnPt.update(obj.tau * obj.nnP.W + (1 - obj.tau) * obj.nnPt.W); % Soft update of the target network
        end
        
    end
    
end