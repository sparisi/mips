function [nextstate, reward, absorb] = puddle_simulator(state, action)

env = puddle_environment();

if nargin == 0

    % Initial random state
    nextstate = rand(2,1);
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

% Transition function
state = state + normrnd(0,0.01,2,1);
switch action
    case 1 % Left
        state(1) = state(1) - env.step;
    case 2 % Right
        state(1) = state(1) + env.step;
    case 3 % Up
        state(2) = state(2) + env.step;
    case 4 % Down
        state(2) = state(2) - env.step;
    otherwise
        error('Unknown action.')
end

nextstate = min(max(state,env.minstate),env.maxstate);
reward = zeros(2,1);

% Distance from the nearest edge of the puddle plus penalty for being far from the goal
reward(1) = puddle_reward_distance(nextstate);

% Time penalty
reward(2) = -1;

% Terminal condition
absorb = norm(nextstate - env.goal) <= env.step;

end


%% Helper function
function reward = puddle_reward_distance(state)

p1 = [0.1 0.75; % Centers of the first puddle
    0.45 0.75];
p2 = [0.45 0.4; % Centers of the second puddle
    0.45 0.8];

radius = 0.1;
factor = 400;

if state(1) > p1(2,1)
    d1 = norm(state' - p1(2,:));
elseif state(1) < p1(1,1)
    d1 = norm(state' - p1(1,:));
else
    d1 = abs(state(2) - p1(1,2));
end

if state(2) > p2(2,2)
    d2 = norm(state' - p2(2,:));
elseif state(2) < p2(1,2)
    d2 = norm(state' - p2(1,:));
else
    d2 = abs(state(1) - p2(1,1));
end

min_distance_from_puddle = min([d1, d2]);
if min_distance_from_puddle <= radius
    reward = - factor * (radius - min_distance_from_puddle);
else
    reward = 0;
end

end
