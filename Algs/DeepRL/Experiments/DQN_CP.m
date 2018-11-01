clear
close all

%% Get problem specification
mdp = CartPole;
episodes_eval = 100;
steps_eval = 1000;
steps_learn = 300;

% Normalization in [-1,1]
range = [mdp.stateLB, mdp.stateUB];
range(2,:) = range(1,:);
range(4,:) = range(3,:);
m = mean(range,2);
range_centered = bsxfun(@minus,range,m);
preprocessS = @(s)bsxfun(@times, bsxfun(@minus,s,m), 1./range_centered(:,2))';

preprocessR = @(r)r;

dimL1 = 30;
