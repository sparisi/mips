function phi = resource_basis_poly(state)

mdp_vars = resource_mdpvariables();
degree = 2;
numfeatures = basis_poly(degree, mdp_vars.nvar_state, 1);

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numfeatures;
    return
end

% Full second degree polynomial
phi = basis_poly(degree, mdp_vars.nvar_state, 1, state);

return
