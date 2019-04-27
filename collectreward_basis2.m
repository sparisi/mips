function phi = collectreward_basis2(sa)
% BFS_POS is a handle for the features of the (x,y) state.
% N_RWD is the number of rewards located in the environment.
% For the reward flags there are no special features.

if nargin == 0
    phi = 9;
    return
end

s = sa(1:end-2,:);
a = sa(end-1:end,:);
phi = [s; a];
