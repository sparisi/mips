function y = zdt6(x)

assert(all(all(x <= 1 & x >= 0)))
[dim, n] = size(x);
assert(dim == 10)
y = zeros(2,n);

g = 1 + 9 * (sum(x(2:dim,:))/(dim-1)).^0.25;

y(1,:) = 1 - exp(-4*x(1,:)) .* sin(6*pi*x(1,:)).^6;
y(2,:) = g .* (1 - (y(1,:)./g).^2);

y = -y;
