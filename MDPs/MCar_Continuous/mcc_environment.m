function env = mcc_environment

env.dt = 0.1;   % Timestep
env.mass = 1;   % Mass
env.g = 9.81;   % Gravity

% Min and max position of the car
env.xLB = -1;
env.xUB = 1;

% Min and max velocity
env.vLB = -3;
env.vUB = 3;

% Min and max acceleration
env.aLB = -4;
env.aUB = 4;

end
