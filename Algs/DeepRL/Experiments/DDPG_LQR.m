clear
close all


%% Get problem specification
mdp = LQR(2);
episodes_eval = 150;
steps_eval = 150;
steps_learn = 150;

% Clip in [-30, -30] and normalization in [-1,1]
mdp.stateLB(:) = -30;
mdp.stateUB(:) = 30;
range = [mdp.stateLB, mdp.stateUB];
m = mean(range,2);
range_centered = bsxfun(@minus,range,m);
preprocessS = @(s)bsxfun(@times, bsxfun(@minus, ...
    bsxfun(@max, bsxfun(@min,s,mdp.stateUB), mdp.stateLB), m), 1./range_centered(:,2))';

% Clip at -300
preprocessR = @(r) (bsxfun(@max, r, -300));

noise_std = 1.2;
