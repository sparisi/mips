function env = cart_environment

env.minstate = [-2.4, -inf, -pi/15, -inf]';
env.maxstate = [2.4, inf, pi/15, inf]';
        
env.g = 9.81;
env.masscart = 1.0;
env.masspole = 0.1;
env.length = 0.5; % Actually distance from the pivot to the pole centre of mass (so full length is 1)
env.force_mag = 10.0;
env.dt = 0.02;

end
