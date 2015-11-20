function [nextstate, reward, absorb] = mcc_simulator(state, action)

if nargin == 0
    
    % Initial state
    nextstate = [-0.5; 0];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

model = mcc_environment;

% Parse input
position = state(1);
velocity = state(2);
acceleration = action;

% Penalty for excessive actions
if acceleration > model.aUB || acceleration < model.aLB 
    reward = -100;
    absorb = true;
    nextstate = state; % The car does not move (too much acceleration -> it explodes)
    return
end

% Update state
psecond = ddp(model, position, velocity, acceleration);
pNext = position + model.dt * velocity + 0.5 * model.dt * model.dt * psecond;
vNext = velocity + model.dt * psecond;
nextstate = [pNext vNext]';

% Penalize if the car exceeds velocity and position limits
if (pNext < model.xLB) || (vNext > model.vUB) || (vNext < model.vLB)
    reward = -100; 
    absorb = true;
    return
end

absorb = position >= model.xUB; % The episode ends when the car reaches the goal
reward = -1; % Time penalty
if absorb, reward = 10; end % Bonus for reaching the goal

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
