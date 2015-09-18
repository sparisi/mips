function phi = deep_basis_poly(state)

numfeatures = 4;

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numfeatures;
    return
end

phi = zeros(numfeatures,1);
phi(1) = 1;
phi(2) = state(1);
phi(3) = state(2);
phi(4) = state(1)*state(2);

return
