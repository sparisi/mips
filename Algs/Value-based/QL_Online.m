% Online Q-learning.
% To update the table of Q-values, only the current single sample is used.
% All previous data is used just to for the evaluation.

clear all
close all

rng(2)

mdp = Gridworld;
% mdp = DeepSeaTreasure;
% mdp = Resource;

robj = 1;
gamma = mdp.gamma;
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
Q = zeros(size(allstates,1),nactions);
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
        [nextstate, reward, terminal] = mpd.simulator(state, action);
        [~, idx_s(end+1)] = ismember(state',allstates,'rows');
        [~, idx_sn(end+1)] = ismember(nextstate',allstates,'rows');
        [~, idx_a(end+1)] = ismember(action',allactions);
        linidx(end+1) = sub2ind(size(Q),idx_s(end),idx_a(end));
        
        % Record sample
        data.a(:,end+1) = action;
        data.r(:,end+1) = reward;
        data.s(:,end+1) = state;
        data.nexts(:,end+1) = nextstate;
        data.terminal(:,end+1) = terminal;
        
        % Continue
        state = nextstate;
        
        % Q-learning
        E = reward(robj) + gamma * max(Q(idx_sn(end),:),[],2)' .* ~terminal - Q(idx_s(end),idx_a(end));
        Q(idx_s(end),idx_a(end)) = Q(idx_s(end),idx_a(end)) + lrate * E;
        
        % Evaluation and plotting
        E = data.r(robj,:) + gamma * max(Q(idx_sn,:),[],2)' .* ~data.terminal - Q(linidx);
        E_history(totsteps) = mean(E.^2);
        
        totsteps = totsteps + 1;

        % Update policy as well
        policy.drawAction = @(s)egreedy( Q(ismember(allstates,s','rows'),:)', epsilon );
        
    end

    updateplot('MS TD Error',totsteps,mean(E.^2),1)
    [V, opt] = max(Q,[],2);
    subimagesc('Q-function',X,Y,Q')
    subimagesc('V-function',X,Y,V')
    subimagesc('Action',X,Y,opt')
    autolayout

end


%% Show
policy_eval.drawAction = @(s)egreedy( Q(ismember(allstates,s','rows'),:)', 0 );
show_simulation(mdp, policy_eval, 1000, 0.05)
