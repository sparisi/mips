function y = quadcost(x, xdes, C)
% Quadratic cost: y = -(x-xdes)'*C*(x-xdes).

if nargin < 3
    d = size(x,1);
    C = eye(d);
    if nargin < 2
        xdes = zeros(d,1);
    end
end

diff = bsxfun(@minus,x,xdes);
y = -sum(bsxfun(@times, diff'*C, diff'), 2)';
