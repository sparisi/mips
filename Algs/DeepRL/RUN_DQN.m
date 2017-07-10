%% Neural networks layout
dimA = mdp.actionUB;
dimO = mdp.dstate;
nnQ = Network([ ...
    Lin(dimO,dimL1) ...
    Bias(dimL1) ...
    ReLU() ...
    Lin(dimL1,dimA) ...
    Bias(dimA) ...
    ]);


%% Gradient descent optimizer
optimQ = RMSprop(numel(nnQ.W));
optimQ.alpha = 0.000025;
optimQ.beta = 0.95;
optimQ.gamma = 0.95;
optimQ.epsilon = 0.01;


%% Learner
learner = DQN_Solver(nnQ,optimQ,dimA,dimO); % Single DQN
% learner = DDQN_Solver(nnQ,optimQ,dimA,dimO); % Double DQN
learner.mdp = mdp;
learner.gamma = mdp.gamma;
learner.preprocessS = preprocessS;
learner.preprocessR = preprocessR;


%% Init learning
decay = 0.995;
episode = 1;

% mdp.showplot
learner.warmup;

if mdp.dstate == 2
density = 10;
Xnodes = linspace(-1,1,density);
Ynodes = linspace(-1,1,density);
[X, Y] = meshgrid(Xnodes,Ynodes);
end


%% Learning
while learner.t < 1e6
    
    policy.drawAction = @(s)argmax( learner.nnQ.forward(preprocessS(s))',1 );
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy);
    [~, episodeLoss, ~] = learner.train();
    
%     learner.epsilon = max(learner.epsilon * decay, 0.1);
    
    updateplot('Expected Return', learner.t, J, 1)
    updateplot('TD Error', learner.t, episodeLoss, 1)
%     updateplot('Epsilon', learner.t, learner.epsilon, 1)
%     updateplot('Q-network Parameters', learner.t, learner.nnQ.W)
    
    if mdp.dstate == 2
    Q_plot = learner.nnQ.forward([X(:)';Y(:)']')';
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
