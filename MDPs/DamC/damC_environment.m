function env = damC_environment()

env.S = 1.0; % reservoir surface
env.W_IRR = 50.0; % water demand
env.H_FLO_U = 50.0; % flooding threshold (upstream)
env.S_MIN_REL = 100.0; % release threshold (i.e., max capacity)
env.DAM_INFLOW = normrnd(40,10); % random inflow (e.g. rain)
env.Q_MEF = 0.0;
env.GAMMA_H2O = 1000.0; % water density
env.W_HYD = 4.36; % hydroelectric demand
env.Q_FLO_D = 30.0; % flooding threshold (downstream)
env.ETA = 1.0; % turbine efficiency
env.G = 9.81; % gravity

return
