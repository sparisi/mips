function y = zdt1(x)

assert(all(all(x <= 1 & x >= 0)))
[dim, n] = size(x);
assert(dim == 30)
y = zeros(2,n);

g = 1 + 9*sum(x(2:dim,:))/(dim-1);

y(1,:) = x(1,:);
y(2,:) = g.*(1-sqrt(x(1,:)./g));

y = -y;