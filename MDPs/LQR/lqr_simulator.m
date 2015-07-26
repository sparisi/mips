function [nextstate, reward, absorb] = lqr_simulator(state, action)

mdp_vars = lqr_mdpvariables();
dim = mdp_vars.dim;
LQR = lqr_environment(dim);

if nargin == 0
    
    nextstate = LQR.x0;
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

absorb = 0;
nextstate = LQR.A*state + LQR.B*action;
reward = zeros(dim,1);
for i = 1 : dim
    reward(i) = -(state'*LQR.Q{i}*state + action'*LQR.R{i}*action); 
end

return

