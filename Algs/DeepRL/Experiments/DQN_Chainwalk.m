clear
close all

%% Get problem specification
mdp = Chainwalk;
episodes_eval = 100;
steps_eval = 100;
steps_learn = 20;

% Normalization in [-1,1]
range = [mdp.stateLB, mdp.stateUB];
m = mean(range,2);
range_centered = bsxfun(@minus,range,m);
preprocessS = @(s)bsxfun(@times, bsxfun(@minus,s,m), 1./range_centered(:,2))';

% Reward is in [-1,0]
preprocessR = @(r)r;

dimL1 = 100;
