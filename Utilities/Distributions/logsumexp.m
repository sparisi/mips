function s = logsumexp(x, dim)
% LOGSUMEXP Computes log(sum(exp(x),dim)) while avoiding numerical 
% underflow.
% By default dim = 1 (columns).
%
% =========================================================================
% ACKNOWLEDGEMENT
% http://www.mathworks.com/matlabcentral/fileexchange/26184-em-algorithm-for-gaussian-mixture-model

if nargin == 1, 
    % Determine which dimension sum to use
    dim = find(size(x)~=1,1);
    if isempty(dim), dim = 1; end
end

% Subtract the largest in each column
y = max(x,[],dim);
x = bsxfun(@minus,x,y);
s = y + log(sum(exp(x),dim));
i = find(~isfinite(y));
if ~isempty(i)
    s(i) = y(i);
end
