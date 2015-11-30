function [nextstate, reward, absorb] = mcc_simulator(state, action)

model = mcc_environment;

if nargin == 0
    
    % Initial state
    nextstate = ([model.xUB; model.vUB] - [model.xLB; model.vLB]) .* rand(2,1) ...
        + [model.xLB; model.vLB];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

% Parse input
position = state(1);
velocity = state(2);
acceleration = action;

% Update state
psecond = ddp(model, position, velocity, acceleration);
pNext = position + model.dt * velocity + 0.5 * model.dt * model.dt * psecond;
vNext = velocity + model.dt * psecond;
nextstate = [pNext vNext]';

% Compute reward
[reward, absorb] = mcar_reward(model, pNext, vNext);

end

%% Helper functions
function dhill_val = dhill(pos)
% Derivative of the hill
if (pos < 0.0)
    dhill_val = 2*pos + 1;
else
    dhill_val = 1 / sqrt(1 + 5*pos^2) - 5*pos^2 / (1 + 5*pos^2)^1.5;
end
end

function ddp_val = ddp(model, pos, velocity, acceleration)
% Second derivative of the hill
A = acceleration / ( model.mass * (1 + dhill(pos) * dhill(pos)) );
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