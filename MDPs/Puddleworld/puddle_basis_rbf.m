function phi = puddle_basis_rbf(state)

env = puddle_environment();
n_centers = 4;
range = [env.minstate, env.maxstate];
numfeatures = basis_krbf(n_centers,range);

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numfeatures + 1;
    return
end

phi = [1; basis_krbf(n_centers,range,state)];

return;
