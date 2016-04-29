function actions = egreedy(Q, epsilon)
% EGREEDY Epsilon-greedy distribution. 
% 
%    INPUT
%     - Q       : [A x N] matrix, where A is the number of possible actions 
%                 and N is the number of samples
%     - epsilon : randomness
%
%    OUTPUT
%     - actions : [1 x N] vector with the actions drawn from the
%                 distribution

[nactions, nsamples] = size(Q);
greedy_action = bsxfun(@ismember,Q,max(Q,[],1));
prob_list = epsilon / nactions * ones(nactions,nsamples);
remainder = bsxfun(@times, (1 - epsilon) * ones(size(prob_list)), 1 ./ sum(greedy_action,1));
prob_list(greedy_action) = prob_list(greedy_action) + remainder(greedy_action);
prob_list = bsxfun(@times, prob_list, 1./sum(prob_list)); % Ensure that the sum is 1
actions = mymnrnd(prob_list,nsamples); % Draw one action for each state
