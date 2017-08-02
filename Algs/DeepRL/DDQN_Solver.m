classdef DDQN_Solver < handle
% Deep Double Q-Learning Network.
%
% =========================================================================
% REFERENCE
% H van Hasselt, A Guez, D Silver
% Deep Reinforcement Learning with Double Q-learning (2015)

    properties

        %% Hyperparameters
        bsize    = 32;   % Minibatch size
        dsize    = 1e5;  % Database size
        minsize  = 1e4;  % Warm up time (number of random samples to collect before learning)
        gamma    = 0.99; % Discount factor
        maxsteps = 1e3;  % Max steps per episode
        epsilon  = 1;    % Parameter of the e-greedy policy
        tau      = 0.1;  % Soft update of the target network

        %% Functions
        mdp                  % MDP with functions for initial state distribution and transition function
        preprocessS = @(s)s; % Preprocess the state
        preprocessR = @(r)r; % Preprocess the reward
        
        %% Networks
        nnQ     % Q-network
        nnQt    % Target network
        optimQ  % Gradient descent optimizer
        
        %% Stored data
        t = 0;  % Total elapsed timesteps
        data    % Replay buffer
        dimA    % Number of actions
        
    end
    
    methods

        %% Constructor
        function obj = DDQN_Solver(nnQ, optimQ, dimA, dimO)
            obj.optimQ      = optimQ;
            obj.dimA        = dimA;
            obj.data.o      = NaN(obj.dsize,dimO); % Observations
            obj.data.a      = NaN(obj.dsize,1); % Actions
            obj.data.r      = NaN(obj.dsize,1); % Rewards
            obj.data.o_next = NaN(obj.dsize,dimO); % Observations of the next state
            obj.data.term   = NaN(obj.dsize,1); % Flag for terminal states
            obj.nnQ         = nnQ;
            obj.nnQt        = copy(nnQ);
        end

        %% Collect random samples before learning
        function warmup(obj)
            while obj.t < obj.minsize
                state = obj.mdp.initstate(1);
                step = 0;
                terminal = false;
                while ~terminal && obj.t < obj.minsize
                    step = step + 1;
                    action = randi(obj.dimA);
                    [nextstate, reward, terminal] = obj.mdp.simulator(state, action);
                    obj.storedata(state, action, reward, nextstate, terminal);
                    terminal = terminal || step == obj.maxsteps;
                    state = nextstate;
                end
            end
        end

        %% Store tuples (s,a,r,s')
        function storedata(obj, state, action, reward, nextstate, terminal)
            obs = obj.preprocessS(state);
            obs_next = obj.preprocessS(nextstate);
            reward = obj.preprocessR(reward);
            
            obj.t = obj.t + 1;
            idx   = mod(obj.t-1,obj.dsize) + 1;
            obj.data.o(idx,:)      = obs;
            obj.data.a(idx,:)      = action;
            obj.data.r(idx,:)      = reward;
            obj.data.o_next(idx,:) = obs_next;
            obj.data.term(idx,:)   = terminal;
        end
        
        %% Training
        function [J, avgL, states] = train(obj)
            assert(obj.t >= obj.minsize, 'Not enough samples in the database.')

            % Get init state and initialize variables
            state = obj.mdp.initstate(1);
            states = zeros(obj.maxsteps, length(state));
            J = 0; % Expected return
            L = []; % Q-function loss
            step = 0;
            terminal = false;

            obj.epsilon = 1;
            
            % Run episode
            while ~terminal
                step = step + 1;
                states(step, :) = state;
                
                obj.epsilon = obj.epsilon * 0.995;

                Q = forward(obj.nnQ, obj.preprocessS(state));
                action = egreedy(Q', obj.epsilon)';

                [nextstate, reward, terminal] = obj.mdp.simulator(state, action);
                obj.storedata(state,action,reward,nextstate,terminal);
                terminal = terminal || step == obj.maxsteps;
                state = nextstate;
                
                J = J + obj.gamma^(step-1)*reward;
                L(end+1) = obj.step();
            end
            
            states = states(1:step,:);
            avgL = mean(L);
        end
        
        function L = step(obj)
            mb = randperm(min(obj.t,obj.dsize),obj.bsize); % Random minibatches
            
            O_next = obj.data.o_next(mb,:);              % bsize x dimO
            O      = obj.data.o(mb,:);                   % bsize x dimO
            Ai     = ind2vec(obj.data.a(mb,:),obj.dimA); % bsize x dimA
            R      = obj.data.r(mb,:);                   % bsize x 1
            Term   = obj.data.term(mb,:);                % bsize x 1

            % Compute targets via Bellman equation with target network
            QT_next     = obj.nnQt.forward(O_next);
            Q_next      = obj.nnQ.forward(O_next);
            [~, A_next] = max(Q_next,[],2);
            T           = R + obj.gamma .* sum(QT_next .* ind2vec(A_next,obj.dimA),2) .* ~Term;
            
            % Compute error, loss and gradient
            Q  = obj.nnQ.forwardfull(O);
            E  = T - sum(Ai.*Q, 2);
            L  = mean(E.^2);
            dL = - 2 * bsxfun(@times, E, Ai) / obj.bsize;
            dW = obj.nnQ.backward(dL);
            
            % Perform gradient descent step and update the Q-network
            obj.nnQ.update(obj.optimQ.step(obj.nnQ.W, dW));
            
            % Soft update of the target network
            obj.nnQt.update(obj.tau * obj.nnQ.W + (1 - obj.tau) * obj.nnQt.W); 
        end

    end

end