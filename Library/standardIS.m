function W = standardIS(target, samplers, Actions, States)
% STANDARDIS Computes importance sampling weights. Each sample ACTIONS(i) 
% is drawn from the distribution SAMPLERS(i).
%
%    INPUT
%     - target   : the distribution to be updated
%     - samplers : the distributions used for sampling
%     - Actions  : [D x N] matrix of parameters (D is the size of the
%                  parameters)
%     - States   : (optional) [S x N] matrix of states (S is the size of 
%                  the state)
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

if nargin == 4
    p = target.evaluate(Actions,States)';
    for i = 1 : N
        q(i) = samplers(i).evaluate(Actions(:,i),States(:,i));
    end
else
    p = target.evaluate(Actions)';
    for i = 1 : N
        q(i) = samplers(i).evaluate(Actions(:,i));
    end
end

W = p ./ q;
W = W';
