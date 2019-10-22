% Batch Q-learning.
% First, data is collected with a random policy. 
% Then, the table of Q-values is updated until convergence using all data.

clear all
close all

mdp = Gridworld;
mdp = GridworldSparse;
% mdp = DeepSeaTreasure;
% mdp = Resource;

robj = 1;
gamma = mdp.gamma;
allstates = mdp.allstates;
X = unique(allstates(:,1));
Y = unique(allstates(:,2));
allactions = 1 : size(mdp.allactions,2);
nactions = length(allactions);

maxepisodes = 10000;
maxsteps = 100;
policy.drawAction = @(s)myunidrnd(mdp.actionLB,mdp.actionUB,size(s,2));
data = collect_samples2(mdp, maxepisodes, maxsteps, policy);


%% Learn
Q = zeros(size(allstates,1),nactions);
lrate = 0.1;

[~, idx_s] = ismember(data.s',allstates,'rows');
[~, idx_sn] = ismember(data.nexts',allstates,'rows');
[~, idx_a] = ismember(data.a',allactions);
linidx = sub2ind(size(Q),idx_s,idx_a);
iter = 1;

while iter < 10000
    E = data.r(robj,:) + gamma * max(Q(idx_sn,:),[],2)' .* ~data.terminal - Q(linidx)';
    E_history(iter) = mean(E.^2);
    Q(linidx) = Q(linidx) + lrate * E';
    
    % Plot
    updateplot('MS TD Error',iter,mean(E.^2),1)
    [V, opt] = max(Q,[],2);
    subimagesc('Q-function',X,Y,Q')
    subimagesc('V-function',X,Y,V')
    subimagesc('Action',X,Y,opt')
    if iter == 1, autolayout, end
    
    iter = iter + 1;
end


%% Show
policy_eval.drawAction = @(s)egreedy( Q(mdp.get_state_idx(s),:)', 0 ); 
show_simulation(mdp, policy_eval, 1000, 0.1)
