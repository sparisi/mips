clear
close all

%% Get problem specification
mdp = DoubleLink;
episodes_eval = 100;
steps_eval = 1000;
steps_learn = 100;

% Normalization in [-1,1]
range = [mdp.stateLB, mdp.stateUB];
m = mean(range,2);
range_centered = bsxfun(@minus,range,m);
preprocessS = @(s) bsxfun(@times, bsxfun(@minus,s,m), 1./range_centered(:,2))';
% preprocessS = @(s) s';

preprocessR = @(r) r;

noise_std = 20;
