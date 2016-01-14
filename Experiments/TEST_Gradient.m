clear all
mdp = DeepSeaTreasure;
robj = 1;
trials = 10;
episodes = 100;
steps = 50;
bfs = @deep_basis_poly;
dreward = mdp.dreward;
gamma = mdp.gamma;
nactions = mdp.actionUB;
pol = Gibbs(bfs, zeros(bfs()*(nactions-1),1),mdp.actionLB:mdp.actionUB);

g0 = zeros(pol.dparams, trials);
g1 = zeros(pol.dparams, trials);
g2 = zeros(pol.dparams, trials);
g3 = zeros(pol.dparams, trials);
g4 = zeros(pol.dparams, trials);
g5 = zeros(pol.dparams, trials);

for i = 1 : trials
    
    ds = collect_samples(mdp, episodes, steps, pol);
    
    X = eREINFORCE(pol,ds,gamma);
    g0(:,i) = X(:,robj);
    X = eREINFORCEbase(pol,ds,gamma);
    g1(:,i) = X(:,robj);
    X = GPOMDP(pol,ds,gamma);
    g2(:,i) = X(:,robj);
    X = GPOMDPbase(pol,ds,gamma);
    g3(:,i) = X(:,robj);
    X = eNAC(pol,ds,gamma);
    g4(:,i) = X(:,robj);
    X = eNACbase(pol,ds,gamma);
    g5(:,i) = X(:,robj);
    
end

stds = [std(g0,1,2), ...
    std(g1,1,2), ...
	std(g2,1,2), ...
	std(g3,1,2), ...
	std(g4,1,2), ...
	std(g5,1,2)];
totstds = sum(stds,1);

clc

delimiter = repmat('-', 1, 11*6-3);
fprintf(['\n ' delimiter '\n'])
fprintf('      REINFORCE      |       GPOMDP        |        eNAC')
fprintf(['\n ' delimiter '\n'])
fprintf(' Plain    | Baseline | Plain    | Baseline | Plain    | Baseline')
fprintfmat(stds, size(stds,1), 1, 6, 'f')
fprintf('                            Total std   ')
fprintfmat(totstds, size(totstds,1), 1, 5, 'f')