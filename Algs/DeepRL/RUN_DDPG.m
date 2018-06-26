dimA = mdp.daction;
dimO = mdp.dstate;


%% Q-network and P-network layout
nnQ = Network([dimO+dimA, 40, 30, 1], {'ReLU', 'ReLU'});
nnP = Network([dimO, 40, 30, dimA], {'ReLU', 'ReLU'});


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
learner = DDPG_Solver(nnP,nnQ,optimP,optimQ,dimA,dimO,mdp);
learner.gamma = mdp.gamma;
learner.maxsteps = steps_learn;
learner.preprocessS = preprocessS;
learner.preprocessR = preprocessR;
learner.sigma = noise_std;


%% Init params
episode = 1;


%% Plotting
% mdp.showplot

if mdp.dstate == 2 % Plot V and Q if state is 2d
density = 10;
Xnodes = linspace(mdp.stateLB(1),mdp.stateUB(1),density);
Ynodes = linspace(mdp.stateLB(2),mdp.stateUB(2),density);
[X, Y] = meshgrid(Xnodes,Ynodes);
XY = learner.preprocessS([X(:)'; Y(:)']);
end


%% Learning
while learner.t < 1e6
    
    if any(isinf(mdp.actionUB))
        policy.drawAction = @(s)learner.nnP.forward(preprocessS(s))';
    else
        policy.drawAction = @(s)bsxfun(@times, learner.nnP.forward(preprocessS(s))', mdp.actionUB);
    end
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy);

    [~, ep_loss] = learner.train();
    updateplot('Expected Return', learner.t, J, 1)
    updateplot('TD Error', learner.t, ep_loss, 1)
%     updateplot('Q-network Parameters', learner.t, learner.nnQ.W)
%     updateplot('P-network Parameters', learner.t, learner.nnP.W)

    % Plotting
    if mdp.dstate == 2
        if any(isinf(mdp.actionUB))
            P_plot = learner.nnP.forward(XY)';
        else
            P_plot = bsxfun(@times, learner.nnP.forward(XY)', mdp.actionUB);
        end
        Q_plot = learner.nnQ.forward([P_plot;XY']')';
        subimagesc('Policy',Xnodes,Ynodes,P_plot)
        subimagesc('V-function - Q(s,pi(s))',Xnodes,Ynodes,Q_plot)
    end
    
    if episode == 1, autolayout, end
    episode = episode + 1;
    
end


%% Show policy
if any(isinf(mdp.actionUB))
    policy.drawAction = @(s)learner.nnP.forward(preprocessS(s))';
else
    policy.drawAction = @(s)bsxfun(@times, learner.nnP.forward(preprocessS(s))', mdp.actionUB);
end
show_simulation(mdp, policy, steps_eval, 0.01)
