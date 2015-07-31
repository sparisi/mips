function phi = mountain_basis_pol(state, action)

mdp_vars = mountain_mdpvariables();

numfeatures = 3;

% The basis functions are repeated for each action but the last one
numbasis = numfeatures * (length(mdp_vars.action_list) - 1);

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numbasis;
    return
end

tmp = zeros(numfeatures,1);
tmp(1) = 1;
tmp(2) = state(1);
tmp(3) = state(2);

% Features depending only on the state
if nargin == 1
    phi = tmp;
    return
end

% Initialize
phi = zeros(numbasis,1);

% Find the starting position
base = numfeatures;

i = action - 1;
phi(base*i+1:base*i+base) = tmp;

return
