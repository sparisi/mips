function phi = puddle_basis_rbf(state)

env = puddle_environment();
n_centers = 4;
range = [env.minstate, env.maxstate];

% If no arguments just return the number of basis functions
if nargin == 0
    phi = basis_krbf(n_centers,range) + 1;
    return
end

assert(size(state,1) == 2);
phi = [ones(1,size(state,2)); basis_krbf(n_centers,range,state)];

return;
