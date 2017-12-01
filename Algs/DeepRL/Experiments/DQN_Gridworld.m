clear
close all

%% Get problem specification
mdp = Gridworld;
episodes_eval = 100;
steps_eval = 20;
steps_learn = 50;

% Normalization in [-1,1]
range = [mdp.stateLB, mdp.stateUB];
m = mean(range,2);
range_centered = bsxfun(@minus,range,m);
preprocessS = @(s)bsxfun(@times, bsxfun(@minus,s,m), 1./range_centered(:,2))';

% Normalization in [-1,1]
preprocessR = @(r)(normalize_data(r,mdp.rewardLB,mdp.rewardUB) - 0.5)*2;

dimL1 = 30;
