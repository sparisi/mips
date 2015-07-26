% Plots the minimum distance from each cell to the edge of a puddle.
% If the cell is not inside the puddle, the distance is 0.
% It also evaluates the average penalty all over the map.

map = zeros(20);
stepsize = 0.05; % environment stepsize
n_cells = 0;
for i = 0 : stepsize : 1
    for j = 0 : stepsize : 1
        state = [i; j];
        dist = puddle_reward_distance(state);
        map(int64(1+i/stepsize),int64(1+j/stepsize)) = dist;
        n_cells = n_cells + 1;
    end
end

avg_J = (-1.7103e+003 - sum(sum(map))) / 399
% 399 because the initial state cannot be the goal position or a cell on 
% the border of the grid
% -sum(sum(map)) because the reward is given only for doing an action and
% this means that at the initial state no penalty is given

mapdist = rot90(map);

% NB: to have such average we need to uniformly sample all possible initial
% states, that are 20x20 = 400. This means that, if we don't want to
% 'cheat' (i.e. use every different initial state for each episode during the
% learning), we need a lot of samples to have a uniform sampling of all the
% initial states.