function W = mixtureIS(target, samplers, N, Actions, States)
% MIXTUREIS Computes importance sampling weights. N_TOT samples are drawn 
% from a uniform mixture of N_TOT / N policies. Each policy collects N 
% batches of samples.
%
%    INPUT
%     - target   : the distribution to be updated
%     - samplers : the distributions used for sampling, assumed to belong
%                  to an uniform mixture of N_TOT / N policies
%     - N        : number of samples drawn by each sampler distribution
%     - Actions  : [D x N_TOT] matrix of parameters (D is the size of the
%                  parameters)
%     - States   : (optional) [S x N_TOT] matrix of states (S is the size 
%                  of the state)
%
%    OUTPUT
%     - W        : [1 x N_TOT] importance sampling weights
%
% =========================================================================
% REFERENCE
% A Owen and Y Zhou
% Safe and effective importance sampling (2000)

N_TOT = size(Actions,2);
Q = zeros(N_TOT, N_TOT); % Q(i,j) = probability of drawing Actions(i) from policy q(j) (q = sampling)

if nargin == 5
    p = target.evaluate(Actions,States)'; % p(i) = probability of drawing Actions(i) from policy p (p = target)
    for j = 1 : N : N_TOT
        L = samplers(j).evaluate(Actions,States)';
        Q(:, j:j+N-1) = repmat(L,1,N);
    end
else
    p = target.evaluate(Actions)';
    for j = 1 : N : N_TOT
        L = samplers(j).evaluate(Actions)';
        Q(:, j:j+N-1) = repmat(L,1,N);
    end
end
Q = Q / N_TOT;
W = p ./ sum(Q,2); % Mixture IS (mixture proportions are equal)
W = W';
