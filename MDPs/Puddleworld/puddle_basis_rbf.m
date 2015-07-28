function phi = puddle_basis_rbf(state, action)

env = puddle_environment();
mdp_vars = puddle_mdpvariables();
n_centers = 10;
n_actions = length(mdp_vars.action_list);
range = [env.xmin,env.xmax;env.ymin,env.ymax];
numfeatures = basis_krbf(n_centers,range);
dim_phi = numfeatures * (n_actions - 1);

% If no arguments just return the number of basis functions
if nargin == 0
    phi = dim_phi;
    return
end

% Full second degree polynomial
phi = basis_krbf(n_centers,range,state);

% Basis depending only on the state
if nargin == 1
    return
end

% Basis depending also on the action
tmp = zeros(dim_phi,1);
base = numfeatures;
i = action - 1;
tmp(base*i+1:base*i+base) = phi;
phi = tmp;

return;
