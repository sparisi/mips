function y = quadcost(x, xdes)
% Quadratic cost: y = -(x-xdes)'*C*(x-xdes)
% The cost matric C is built such that its eigenvalues are linearly chosen
% in [0.1, 1].

if nargin == 1, xdes = 4*ones(size(x,1),1); end

n = size(x,1);
D = diag(linspace(0.1,1,n));
V = orth(magic(n));
C = V*D*V';
y = -diag(bsxfun(@minus,x,xdes)'*C*bsxfun(@minus,x,xdes))';
