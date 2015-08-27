function phi = deep_basis_tile1(state, action)

mdp_vars = deep_mdpvariables();

numfeatures = 4;

% The basis functions are repeated for each action but the last one
numbasis = numfeatures * (length(mdp_vars.action_list) - 1);

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numbasis;
    return
end

tmp = zeros(numfeatures,1);
tmp(1) = 1;
tmp(2) = (state(1) == 1 && state(2) == 1); % First cell
tmp(3) = (state(1) == 1 && state(2) ~= 1 && state(2) ~= 10); % First row but first and last column
tmp(4) = (state(2) == 10); % Last column

% The idea is that the first tile distinguishes between the objectives (if 
% I want to minimize the time, from the first cell I have to go down, right
% otherwise), while the other two are activated when optimizing the 
% treasure (the agent will follow the first row and then the last column). 
% Mixing the parameters corresponding to the first tile we can obtain a
% policy which is a mixture of the optimal ones.

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
