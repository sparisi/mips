function actions = softmax(Q, tau)
% SOFTMAX Softmax distribution. 
% 
%    INPUT
%     - Q       : [A x N] matrix, where A is the number of possible actions 
%                 and N is the number of samples
%     - tau     : temperature
%
%    OUTPUT
%     - actions : [1 x N] vector with the actions drawn from the
%                 distribution

tau = max(tau,1e-8); % Avoid numerical problems with tau = 0
[~, nsamples] = size(Q);
Q = bsxfun(@minus, Q, max(Q,[],1)); % Numerical trick to avoid Inf and NaN (the distribution does not change)
exp_term = exp(Q / tau);
prob_list = bsxfun(@times, exp_term, 1./sum(exp_term,1));
prob_list = bsxfun(@times, prob_list, 1./sum(prob_list)); % Ensure that the sum is 1
actions = mymnrnd(prob_list,nsamples); % Draw one action for each sample
