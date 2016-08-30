function y = zdt4(x)

assert(all(x(1,:) <= 1 & x(1,:) >= 0))
assert(all(all(x(2:end,:) <= 5 & x(2:end,:) >= -5)))
[dim, n] = size(x);
assert(dim == 10)
y = zeros(2,n);

g = 1 + 10*(10-1) + sum(x(2:end,:).^2 - 10*cos(4*pi*x(2:end,:)));

y(1,:) = x(1,:);
y(2,:) = g .* (1-sqrt(x(1,:)./g));

y = -y;