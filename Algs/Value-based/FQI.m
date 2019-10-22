% FQI Fitted Q-iteration without approximation.
%
% =========================================================================
% REFERENCE
% D Ernst, P Geurts, L Wehenkel
% Tree-based batch mode reinforcement learning (2005)

clear all
close all

mdp = Gridworld;
mdp = GridworldSparse;
% mdp = DeepSeaTreasure;
% mdp = Resource;

robj = 1;
gamma = mdp.gamma;
gamma = min(gamma,0.999999);
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

[~, idx_s] = ismember(data.s',allstates,'rows');
[~, idx_sn] = ismember(data.nexts',allstates,'rows');
[~, idx_a] = ismember(data.a',allactions);
linidx = sub2ind(size(Q),idx_s,idx_a);
iter = 1;

while iter < 10000
    T = data.r(robj,:) + gamma * max(Q(idx_sn,:),[],2)' .* ~data.terminal;
    E = T - Q(linidx)';
    Q(linidx) = T;
    
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
show_simulation(mdp, policy_eval, 100, 0.1)
