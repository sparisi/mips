function phi = deep_basis_tile2(state, action)

mdp_vars = deep_mdpvariables();

numfeatures = 11;

% The basis functions are repeated for each action but the last one
numbasis = numfeatures * (length(mdp_vars.action_list) - 1);

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numbasis;
    return
end

% This is a more generic tile coding, with each tile being a 4x4 cell. Each
% tile has the border cells overlapping with its neighbors. In addition,
% there is a 1x1 tile corresponding to the first cell.
% The last row (10) is not covered, as there is only one blank cell and the
% optimal action is to always go down.

tmp = zeros(numfeatures,1);
tmp(1) = 1;
tmp(2) = ( (state(1) >= 1 && state(1) <= 4) && ...
    (state(2) >= 1 && state(2) <= 4) && ~(state(1) == 1 && state(2) == 1) );
tmp(3) = ( (state(1) >= 4 && state(1) <= 7) && ...
    (state(2) >= 1 && state(2) <= 4) );
tmp(4) = ( (state(1) >= 7 && state(1) <= 10) && ...
    (state(2) >= 1 && state(2) <= 4) );
tmp(5) = ( (state(1) >= 1 && state(1) <= 4) && ...
    (state(2) >= 4 && state(2) <= 7) );
tmp(6) = ( (state(1) >= 4 && state(1) <= 7) && ...
    (state(2) >= 4 && state(2) <= 7) );
tmp(7) = ( (state(1) >= 7 && state(1) <= 10) && ...
    (state(2) >= 4 && state(2) <= 7) );
tmp(8) = ( (state(1) >= 1 && state(1) <= 4) && ...
    (state(2) >= 7 && state(2) <= 10) );
tmp(9) = ( (state(1) >= 4 && state(1) <= 7) && ...
    (state(2) >= 7 && state(2) <= 10) );
tmp(10) = ( (state(1) >= 7 && state(1) <= 10) && ...
    (state(2) >= 7 && state(2) <= 10) );
tmp(11) = ( state(1) == 1 && state(2) == 1 );

% Features depending only on the state
if nargin == 1
    phi = tmp;
    return
end

phi = zeros(numbasis,1);

init_idx = numfeatures;

i = action - 1;
phi(init_idx*i+1:init_idx*i+init_idx) = tmp;

return
