clear
close all

%% Get problem specification
mdp = Puddleworld;
episodes_eval = 150;
steps_eval = 50;

% Normalization in [-1,1]
range = [mdp.stateLB, mdp.stateUB];
m = mean(range,2);
range_centered = bsxfun(@minus,range,m);
preprocessS = @(s)bsxfun(@times, bsxfun(@minus,s,m), 1./range_centered(:,2))';

% Normalization in [0,1]
preprocessR = @(r)r(1,:)/400 - 2e-2;

dimL1 = 30;
