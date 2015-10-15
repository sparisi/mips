function [nextstate, reward, absorb] = puddle2_simulator(state, action)

env = puddle2_environment();

if nargin < 1

    nextstate = rand(2,1);
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

action = min(action, env.step);
action = max(action, -env.step);
nextstate = state + action + normrnd(0,0.01,2,1);
nextstate = min(max(nextstate,env.minstate),env.maxstate);

% Distance from the nearest edge of the puddle plus penalty for being far from the goal
reward = puddle2_reward_distance(nextstate) - 0.1*norm(nextstate - env.goal);

% Terminal condition
if norm(nextstate - env.goal) <= env.step
    absorb = 1;
else
    absorb = 0;
end

end


%% Helper function
function reward = puddle2_reward_distance(state)

p1 = [0.1 0.75; % Centers of the first puddle
    0.45 0.75];
p2 = [0.45 0.4; % Centers of the second puddle
    0.45 0.8];
p3 = [0.8 0.2;  % Centers of the third puddle
    0.8 0.5];
p4 = [0.7 0.75; % Centers of the fourth puddle
    0.7 0.8];

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

if state(2) > p3(2,2)
    d3 = norm(state' - p3(2,:));
elseif state(2) < p3(1,2)
    d3 = norm(state' - p3(1,:));
else
    d3 = abs(state(1) - p3(1,1));
end

if state(2) > p4(2,2)
    d4 = norm(state' - p4(2,:));
elseif state(2) < p4(1,2)
    d4 = norm(state' - p4(1,:));
else
    d4 = abs(state(1) - p4(1,1));
end

min_distance_from_puddle = min([d1, d2, d3, d4]);
if min_distance_from_puddle <= radius
    reward = - factor * (radius - min_distance_from_puddle);
else
    reward = 0;
end

end
