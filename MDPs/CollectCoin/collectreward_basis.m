function phi = collectreward_basis(bfs_pos, n_rwd, state)
% BFS_POS is a handle for the features of the (x,y) state.
% N_RWD is the number of rewards located in the environment.
% For the reward flags there are no special features.

if nargin == 2
    phi = bfs_pos() + n_rwd;
    return
end

phi = [bfs_pos(state(1:end-n_rwd,:)); state(end-n_rwd+1:end,:)];
