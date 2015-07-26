function phi = deep_basis_pol_v2(state, action)

mdp_vars = deep_mdpvariables();

numfeatures = 4;

% The polynomial is repeated for each action
numbasis = numfeatures * (length(mdp_vars.action_list) - 1);

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numbasis;
    return
end

tmp = zeros(numfeatures,1);
tmp(1) = 1;
tmp(2) = (state(1) == 1 && state(2) == 1);
tmp(3) = (state(1) == 1 && ~(state(1) == 1 && state(2) == 1) ...
        && ~(state(1) == 1 && state(2) == 10));
tmp(4) = (state(2) == 10);

% Features depending only from the state
if nargin == 1
    phi = tmp;
    return
end

phi = zeros(numbasis,1);

init_idx = numfeatures;

i = action - 1;
phi(init_idx*i+1:init_idx*i+init_idx) = tmp;

return
