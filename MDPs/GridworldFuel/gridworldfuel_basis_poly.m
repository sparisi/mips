function phi = gridworldfuel_basis_poly(state)

numfeatures = 14;

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numfeatures;
    return
end

[d,n] = size(state);
assert(d == 3);
phi = zeros(numfeatures,n);
phi(1,:) = state(1,:);
phi(2,:) = state(2,:);
phi(3,:) = state(3,:);
phi(4:14,:) = mymvnrnd(1:11,eye(11),n)';

return
