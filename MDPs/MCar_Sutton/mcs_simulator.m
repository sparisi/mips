function [nextstate, reward, absorb] = mcs_simulator(state, action)

if nargin == 0
    
    % Initial state
    nextstate = [-0.5; 0];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

model = mcs_environment;

% Model transition
position = state(1);
velocity = state(2);
nextVel = velocity + model.acceleration(action) - cos(model.s * position) * model.g;
nextVel = max(min(nextVel,model.vUB),model.vLB);
nextPos = position + nextVel;
nextPos = max(nextPos,model.xLB);
nextstate = [nextPos; nextVel];

% Number of acceleration actions
reward1 = 0;
% Number of reversing actions
reward2 = 0;

if action == 3
    reward1 = -1;
elseif action == 1
    reward2 = -1;
end

% Time
reward3 = -1;
if nextPos >= model.xUB;
    absorb = 1;
else
    absorb = 0;
end
reward = [reward1; reward2; reward3];

return
