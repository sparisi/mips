clear all
domain = 'deep';
robj = 1;
trials = 5;
episodes = 100;

[N, pol, ~, steps, gamma] = feval([domain '_settings']);

g0 = zeros(pol.dlogPidtheta, trials);
g1 = zeros(pol.dlogPidtheta, trials);
g2 = zeros(pol.dlogPidtheta, trials);
g3 = zeros(pol.dlogPidtheta, trials);
g4 = zeros(pol.dlogPidtheta, trials);
g5 = zeros(pol.dlogPidtheta, trials);

for i = 1 : trials
    
    ds = collect_samples(domain, episodes, steps, pol);
    
    g0(:,i) = eREINFORCE(pol,ds,gamma,robj);
    g1(:,i) = eREINFORCEbase(pol,ds,gamma,robj);
    g2(:,i) = GPOMDP(pol,ds,gamma,robj);
    g3(:,i) = GPOMDPbase(pol,ds,gamma,robj);
    g4(:,i) = eNAC(pol,ds,gamma,robj);
    g5(:,i) = eNACbase(pol,ds,gamma,robj);
    
end

stds = [std(g0,1,2), ...
    std(g1,1,2), ...
	std(g2,1,2), ...
	std(g3,1,2), ...
	std(g4,1,2), ...
	std(g5,1,2)];

clc

delimiter = repmat('-', 1, 11*6-3);
fprintf(['\n ' delimiter '\n'])
fprintf('      REINFORCE      |       GPOMDP        |        eNAC')
fprintf(['\n ' delimiter '\n'])
fprintf(' Plain    | Baseline | Plain    | Baseline | Plain    | Baseline')
fprintfmat(stds, size(stds,1), 1, 6, 'f')