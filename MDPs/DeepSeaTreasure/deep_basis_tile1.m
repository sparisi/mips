function phi = deep_basis_tile1(state)

numfeatures = 3;

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numfeatures;
    return
end

[d,n] = size(state);
assert(d == 2);
phi = zeros(numfeatures,n);

phi(1,:) = (state(1,:) == 1 & state(2,:) == 1); % First cell
phi(2,:) = (state(1,:) == 1 & state(2,:) ~= 1 & state(2,:) ~= 10); % First row but first and last column
phi(3,:) = (state(2,:) == 10); % Last column

return
