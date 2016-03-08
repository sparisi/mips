function W = standardIS(target, samplers, Actions, N)
% STANDARDIS Computes importance sampling weights. Each sample ACTIONS(i) 
% is drawn from the distribution SAMPLERS(i).
%
%    INPUT
%     - target   : the distribution to be updated
%     - samplers : the distributions used for sampling
%     - Actions  : [D x N_TOT] matrix of parameters (D is the size of the
%                  parameters)
%     - N        : number of samples
%
%    OUTPUT
%     - W        : importance sampling weights
%
% =========================================================================
% REFERENCE
% A Owen and Y Zhou
% Safe and effective importance sampling (2000)

q = zeros(N, 1);
p = target.evaluate(Actions)';
for i = 1 : N
    q(i) = samplers(i).evaluate(Actions(:,i));
end
W = p ./ q;
