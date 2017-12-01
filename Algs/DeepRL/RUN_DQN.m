rng(1)


%% Neural networks layout
dimA = mdp.actionUB;
dimO = mdp.dstate;

nnQ = Network([dimO, dimL1, dimA], {'ReLU'});


%% Gradient descent optimizer
optimQ = RMSprop(numel(nnQ.W));
optimQ.alpha = 0.00025;
% optimQ.beta = 0.95;
% optimQ.gamma = 0.95;
% optimQ.epsilon = 0.01;


%% Learner
learner = DQN_Solver(nnQ,optimQ,dimA,dimO); % Single DQN
% learner = DDQN_Solver(nnQ,optimQ,dimA,dimO); % Double DQN
learner.mdp = mdp;
learner.gamma = mdp.gamma;
learner.maxsteps = steps_learn;
learner.preprocessS = preprocessS;
learner.preprocessR = preprocessR;


%% Init learning
episode = 1;

% mdp.showplot
learner.warmup;

if mdp.dstate == 1 % Plot V and Q if state is 1d
Xnodes = linspace(mdp.stateLB(1),mdp.stateUB(1),100);
X = learner.preprocessS(Xnodes(:)');
elseif mdp.dstate == 2 % Plot V and Q if state is 2d
density = 10;
Xnodes = linspace(mdp.stateLB(1),mdp.stateUB(1),density);
Ynodes = linspace(mdp.stateLB(2),mdp.stateUB(2),density);
[X, Y] = meshgrid(Xnodes,Ynodes);
XY = learner.preprocessS([X(:)'; Y(:)']);
end


%% Learning
while learner.t < 1e6
    
    policy.drawAction = @(s)argmax( learner.nnQ.forward(preprocessS(s))',1 );
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy);
    [~, ep_loss, ~] = learner.train();
    
    updateplot('Expected Return', learner.t, J, 1)
    updateplot('TD Error', learner.t, ep_loss, 1)
%     updateplot('Epsilon', learner.t, learner.epsilon, 1)
%     updateplot('Q-network Parameters', learner.t, learner.nnQ.W)
    
    % Plotting
    if mdp.dstate == 1
    Q_plot = learner.nnQ.forward(X)';
    V_plot = max(Q_plot,[],1);
    fig = findobj('type','figure','name','V-function');
    if isempty(fig), fig = figure('name','V-function'); plot(X,V_plot); end
    fig.Children.Children.YData = V_plot; drawnow limitrate
    elseif mdp.dstate == 2
    Q_plot = learner.nnQ.forward(XY)';
    V_plot = max(Q_plot,[],1);
    subimagesc('Q-function',Xnodes,Ynodes,Q_plot)
    subimagesc('V-function',Xnodes,Ynodes,V_plot)
    end
    
    if episode == 1, autolayout, end
    episode = episode + 1;
    
end


%% Show policy
policy.drawAction = @(s)argmax( learner.nnQ.forward(preprocessS(s))',1 );
show_simulation(mdp, policy, steps_eval, 0.01)
