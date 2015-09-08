function [ n_obj, policy, episodes, steps, gamma ] = puddle_settings

mdp_vars = puddle_mdpvariables();
n_obj = mdp_vars.nvar_reward;
gamma = mdp_vars.gamma;

bfs = @puddle_basis_tile;
policy = gibbs(bfs, zeros(bfs(),1), mdp_vars.action_list);

episodes = 2000;
steps = 50;

end
