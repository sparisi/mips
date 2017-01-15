clear
close all

%% Get problem specification
mdp = DeepSeaTreasure;
episodes_eval = 1;
steps_eval = 25;

% Normalization in [-1,1]
range = [mdp.stateLB, mdp.stateUB];
m = mean(range,2);
range_centered = bsxfun(@minus,range,m);
preprocessS = @(s)bsxfun(@times, bsxfun(@minus,s,m), 1./range_centered(:,2))';

% Normalization in [0,1]
preprocessR = @(r)r(1,:)/123;

dimL1 = 30;
