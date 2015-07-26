function [nextState, reward, absorb] = mce_simulator(state, action)

if nargin == 0
    
    % Initial state
    nextState = [-0.5; 0];
    return
    
elseif nargin == 1
    
    nextState = state;
    return
    
end

model = mce_environment;

% Parse input
position = state(1);
velocity = state(2);
throttle = model.throttle(action);

% Update state
psecond = ddp(model, position, velocity, throttle);

pNext = position + model.dt * velocity + 0.5 * model.dt * model.dt * psecond;
vNext = velocity + model.dt * psecond;
nextState = [pNext vNext]';

% Compute reward
[reward, absorb] = mcar_reward(model, pNext, vNext);

end

%% Helper functions
function hill_val = hill(pos)
% Equation of the hill
if (pos < 0.0)
    hill_val = pos.^2 + pos;
else
    hill_val = pos ./ sqrt(1 + 5*pos.^2);
end
end

function dhill_val = dhill(pos)
% Derivative of the hill
if (pos < 0.0)
    dhill_val = 2*pos + 1;
else
    dhill_val = 1 / sqrt(1 + 5*pos^2) - 5*pos^2 / (1 + 5*pos^2)^1.5;
end
end

function ddp_val = ddp(model, pos, velocity, throttle)
% Second derivative of the hill
A = throttle / ( model.mass * (1 + dhill(pos) * dhill(pos)) );
B = model.g * dhill(pos) / ( 1 + dhill(pos) * dhill(pos) );
C = velocity^2 * dhill(pos) * dhill(dhill(pos)) / (1 + dhill(pos)^2);
ddp_val = A - B - C;
end

function [reward, absorb] = mcar_reward(model, position, velocity)
% Reward function
reward = 0;
absorb = false;
if (position < model.xLB) || (abs(velocity) > model.vUB)
    reward = -1;
    absorb = true;
elseif (position > model.xUB) && (abs(velocity) <= model.vUB)
    reward = 1;
    absorb = true;
end
end