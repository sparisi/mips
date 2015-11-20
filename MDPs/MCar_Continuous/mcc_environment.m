function env = mcc_environment

env.dt = 0.1;   % Timestep
env.mass = 1;   % Mass
env.g = 9.81;   % Gravity

% Min and max position of the car
env.xLB = -2;
env.xUB = 1;

% Min and max velocity
env.vLB = -4;
env.vUB = 4;

% Min and max acceleration
env.aLB = -4;
env.aUB = 4;

end
