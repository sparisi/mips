function phi = resource_basis_pol(state, action)

mdp_vars = resource_mdpvariables();
n_actions = length(mdp_vars.action_list);
degree = 2;
numfeatures = basis_poly(degree, mdp_vars.nvar_state, 1);
dim_phi = numfeatures * (n_actions - 1);

%%% If no arguments just return the number of basis functions
if nargin == 0
    phi = dim_phi;
    return
end

%%% Full second degree polynomial
phi = basis_poly(degree, mdp_vars.nvar_state, 1, state);

%%% Basis depending only on the state
if nargin == 1
    return
end

%%% Basis depending also on the action
tmp = zeros(dim_phi,1);
base = numfeatures;
i = action - 1;
tmp(base*i+1:base*i+base) = phi;
phi = tmp;

return
