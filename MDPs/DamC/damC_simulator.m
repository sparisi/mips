function [nextstate, reward, absorb] = damC_simulator(state, action, context)

env = damC_environment();

if nargin == 0
    
    nextstate = unifrnd(0,300);
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

% Bound the action
min_action = max(state - context(1), 0);
max_action = state;
penalty = 0;

if min_action > action || max_action < action

    % Penalize for unfeasible actions
    penalty = -max(action - max_action, min_action - action);
    action  = max(min_action, min(max_action, action));
    
end

% Transition dynamic
nextstate = state + normrnd(context(2),10) - action;

% Cost due to the excess level w.r.t. a flooding threshold (upstream)
reward(1) = -max(nextstate/context(3) - context(4), 0) + penalty;

% Deficit in the water supply w.r.t. the water demand
reward(2) = -max(context(7) - action, 0) + penalty;

q = max(action - env.Q_MEF, 0);
p_hyd = context(5) * env.G * env.GAMMA_H2O * nextstate/env.S * q / (3.6e6);

% Deficit in the hydroelectric supply w.r.t the hydroelectric demand
reward(3) = -max(context(6) - p_hyd, 0) + penalty;

% Cost due to the excess level w.r.t. a flooding threshold (downstream)
reward(4) = -max(action - context(8), 0) + penalty;

absorb = 0;

reward = reward(1) + reward(2) + reward(3);

return
