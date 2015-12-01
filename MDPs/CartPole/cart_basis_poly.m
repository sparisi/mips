function phi = cart_basis_poly(state)

mdp_vars = cart_mdpvariables();
degree = 2;
numfeatures = basis_poly(degree, mdp_vars.nvar_state, 1);

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numfeatures;
    return
end

assert(size(state,1) == mdp_vars.nvar_state)

% Full second degree polynomial
phi = basis_poly(degree, mdp_vars.nvar_state, 1, state);

end