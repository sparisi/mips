function y = quadcost(x, xdes, C)
% Quadratic cost: y = -(x-xdes)'*C*(x-xdes).

diff = bsxfun(@minus,x,xdes);
y = -sum(bsxfun(@times, diff'*C, diff'), 2)';
