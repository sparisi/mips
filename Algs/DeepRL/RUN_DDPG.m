dimA = mdp.daction;
dimO = mdp.dstate;


%% Q-networks layout
nnQ = Network([ ...
    Lin(dimA+dimO,40) ...
    Bias(40) ...
    ReLU() ...
    Lin(40,30) ...
    Bias(30) ...
    ReLU() ...
    Lin(30,1) ...
    Bias(1) ...
    ]);


%% Policy networks layout
nnP = Network([ ...
    Lin(dimO,40) ...
    Bias(40) ...
    ReLU() ...
    Lin(40,30) ...
    Bias(30) ...
    Lin(30,dimA) ...
    Bias(dimA) ...
    ]);


%% Gradient descent optimizers
optimP = RMSprop(numel(nnP.W));
optimP.alpha = 0.000025;
optimP.beta = 0.95;
optimP.gamma = 0.95;
optimP.epsilon = 0.01;

optimQ = RMSprop(numel(nnQ.W));
optimQ.alpha = 0.000025;
optimQ.beta = 0.95;
optimQ.gamma = 0.95;
optimQ.epsilon = 0.01;


%% Learner
learner = DDPG_Solver(nnP,nnQ,optimP,optimQ,dimA,dimO);
learner.mdp = mdp;
learner.gamma = mdp.gamma;
learner.preprocessS = preprocessS;
learner.preprocessR = preprocessR;
learner.sigma = noise_std;


%% Init params
decay = 0.995;
episode = 1;


%% Plotting
mdp.showplot

if mdp.dstate == 2
density = 10;
Xnodes = linspace(-1,1,density);
Ynodes = linspace(-1,1,density);
[X, Y] = meshgrid(Xnodes,Ynodes);
end


%% Learning
while learner.t < 1e6
    
    policy.drawAction = @(s)learner.nnP.forward(preprocessS(s))';
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy);

    learner.sigma = learner.sigma * decay;
    [~, episodeLoss] = learner.train();
    updateplot('Expected Return', learner.t, J, 1)
    updateplot('TD Error', learner.t, episodeLoss, 1)
%     updateplot('Q-network Parameters', learner.t, learner.nnQ.W)
%     updateplot('P-network Parameters', learner.t, learner.nnP.W)

    if mdp.dstate == 2
    P_plot = learner.nnP.forward([X(:)';Y(:)']')';
    Q_plot = learner.nnQ.forward([P_plot;X(:)';Y(:)']')';
    subimagesc('Policy',Xnodes,Ynodes,P_plot)
    subimagesc('Q-function',Xnodes,Ynodes,Q_plot)
    end
    
    if episode == 1, autolayout, end
    episode = episode + 1;
    
end


%% Show policy
policy.drawAction = @(s)learner.nnP.forward(preprocessS(s))';
show_simulation(mdp, policy, 50, 0.01)
