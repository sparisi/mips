function env = mcs_environment

% Slope and gravity
env.s   = 3;
env.g   = 0.0025;

% Min and max position of the car
env.xLB = -1.2;
env.xUB = 0.6;

% Min and max velocity
env.vLB = -0.07;
env.vUB = 0.07;

% Actions
env.acceleration = 0.1 * [-1 0 1]';

end
