function y = noisysphere(x)

persistent M
[dim, N] = size(x);
if isempty(M), M = rand(dim); end
y = -sum((M*x).*x) -mymvnrnd(0,1,N).*abs(sum((M*x).*x));
