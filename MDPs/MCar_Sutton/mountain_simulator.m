function [nextstate, reward, absorb] = mountain_simulator(state, action)

if nargin == 0
    
    % Initial state
    nextstate = [-0.5; 0];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

action = action - 2;

vmax = 0.07;
xmin = -1.2;
xmax = 0.6;
accel = 0.001;
gravity = -0.0025;
slope = 3;

% Parse input and update state
v = state(2) + action * accel + cos(slope * state(1)) * gravity;
x = state(1) + v;

v = min(v,vmax);
v = max(v,-vmax);
x = max(x,xmin);

nextstate = [x;v];

% Number of acceleration actions
reward1 = 0;
% Number of reversing actions
reward2 = 0;

if action == 1
    reward1 = -1;
elseif action == -1
    reward2 = -1;
end

% Time
reward3 = -1;
if x >= xmax
    absorb = 1;
else
    absorb = 0;
end
reward = [reward1; reward2; reward3];

return
