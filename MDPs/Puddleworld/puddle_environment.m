function env = puddle_environment()

env.minstate = [0 0]';
env.maxstate = [1 1]';
env.goal = [1 1]';
env.step = [0.05, 0.05]';

return
