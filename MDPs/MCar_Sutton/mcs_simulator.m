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

reward1 = -(action == 3); % Number of acceleration actions
reward2 = -(action == 1); % Number of reversing actions
reward3 = -1; % Time
reward = [reward1; reward2; reward3];
absorb = nextPos >= model.xUB;

return
