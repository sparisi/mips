% Online Double Q-learning.
% To update the table of Q-values, only the current single sample is used.
% All previous data is used just to for the evaluation.
%
% =========================================================================
% REFERENCE
% H van Hasselt
% Double Q-Learing (2010)

clear all
close all

rng(2)

mdp = Gridworld;
% mdp = DeepSeaTreasure;
% mdp = Resource;

robj = 1;
gamma = mdp.gamma;
simulator = @mdp.simulator;
allstates = mdp.allstates;
X = unique(allstates(:,1));
Y = unique(allstates(:,2));
allactions = 1 : size(mdp.allactions,2);
nactions = length(allactions);

data.s = nan(mdp.dstate, 0);
data.nexts = nan(mdp.dstate, 0);
data.a = nan(mdp.daction, 0);
data.r = nan(mdp.dreward, 0);
data.terminal = nan(1, 0);
Q1 = zeros(size(allstates,1),nactions);
Q2 = zeros(size(allstates,1),nactions);
idx_s = [];
idx_sn = [];
idx_a = [];
linidx = [];
lrate = 0.1;
policy.drawAction = @(s)myunidrnd(mdp.actionLB,mdp.actionUB,size(s,2));


%% Collect data and learn
maxepisodes = 10000;
maxsteps = 100;
totsteps = 1;
epsilon = 0.2;

for episode = 1 : maxepisodes
    step = 0;
    state = mdp.initstate(1);
    terminal = 0;

    % Animation + print counter
    disp(episode)
    mdp.showplot
    
    % Run the episodes until maxsteps or terminal state
    while (step < maxsteps) && ~terminal
        
        step = step + 1;
        action = policy.drawAction(state);
        
        % Simulate one step
        [nextstate, reward, terminal] = feval(simulator, state, action);
        [~, idx_s(end+1)] = ismember(state',allstates,'rows');
        [~, idx_sn(end+1)] = ismember(nextstate',allstates,'rows');
        [~, idx_a(end+1)] = ismember(action',allactions);
        linidx(end+1) = sub2ind(size(Q1),idx_s(end),idx_a(end));
        
        % Record sample
        data.a(:,end+1) = action;
        data.r(:,end+1) = reward;
        data.s(:,end+1) = state;
        data.nexts(:,end+1) = nextstate;
        data.terminal(:,end+1) = terminal;
        
        % Continue
        state = nextstate;
        
        if rand < 0.5 % Update Q1
            E = reward(robj) + gamma * max(Q2(idx_sn(end),:),[],2)' .* ~terminal - Q1(idx_s(end),idx_a(end));
            Q1(idx_s(end),idx_a(end)) = Q1(idx_s(end),idx_a(end)) + lrate * E;
        else % Update Q2
            E = reward(robj) + gamma * max(Q1(idx_sn(end),:),[],2)' .* ~terminal - Q2(idx_s(end),idx_a(end));
            Q2(idx_s(end),idx_a(end)) = Q2(idx_s(end),idx_a(end)) + lrate * E;
        end
        
%         % Evaluation and plotting over the avg of the two Q
        Qavg = (Q1 + Q2) / 2;
        E = data.r(robj,:) + gamma * max(Qavg(idx_sn,:),[],2)' .* ~data.terminal - Qavg(linidx);
        E_history(totsteps) = mean(E.^2);
%         updateplot('MS TD Error',iter,mean(E.^2),1)
%         [V, opt] = max(Qavg,[],2);
%         subimagesc('Q-function',X,Y,Qavg')
%         subimagesc('V-function',X,Y,V')
%         subimagesc('Action',X,Y,opt')
%         if iter == 1, autolayout, end
        
        totsteps = totsteps + 1;
        
        % Next action will be chosen using either Q1 or Q2
        if rand < 0.5
            policy.drawAction = @(s)egreedy( Q1(ismember(allstates,s','rows'),:)', epsilon );
        else
            policy.drawAction = @(s)egreedy( Q2(ismember(allstates,s','rows'),:)', epsilon );
        end
        
    end
    
end


%% Show
Qavg = (Q1 + Q2) / 2;
policy_eval.drawAction = @(s)egreedy( Qavg(ismember(allstates,s','rows'),:)', 0 );
show_simulation(mdp, policy_eval, 1000, 0.1)
