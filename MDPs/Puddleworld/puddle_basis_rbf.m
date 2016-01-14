function phi = puddle_basis_rbf(state)

n_centers = 4;
range = [0 1; 0 1];

% If no arguments just return the number of basis functions
if nargin == 0
    phi = basis_krbf(n_centers,range);
else
    assert(size(state,1) == 2);
    phi = basis_krbf(n_centers,range,state);
end
