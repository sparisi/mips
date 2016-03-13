function W = standardIS(target, samplers, Actions)
% STANDARDIS Computes importance sampling weights. Each sample ACTIONS(i) 
% is drawn from the distribution SAMPLERS(i).
%
%    INPUT
%     - target   : the distribution to be updated
%     - samplers : the distributions used for sampling
%     - Actions  : [D x N] matrix of parameters (D is the size of the
%                  parameters)
%
%    OUTPUT
%     - W        : [1 x N] importance sampling weights
%
% =========================================================================
% REFERENCE
% A Owen and Y Zhou
% Safe and effective importance sampling (2000)

N = size(Actions,2);
q = zeros(N, 1);
p = target.evaluate(Actions)';
for i = 1 : N
    q(i) = samplers(i).evaluate(Actions(:,i));
end
W = p ./ q;
W = W';
