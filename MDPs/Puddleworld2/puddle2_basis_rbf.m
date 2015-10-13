function phi = puddle2_basis_rbf(state)

env = puddle2_environment();
n_centers = 10;
range = [env.minstate, env.maxstate];

% If no arguments just return the number of basis functions
if nargin == 0
    phi = basis_krbf(n_centers,range);
else
    assert(size(state,1) == 2);
    phi = [basis_krbf(n_centers,range,state)];
end

return;
