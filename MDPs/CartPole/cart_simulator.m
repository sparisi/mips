function [nextstate, reward, absorb] = cart_simulator(state, action)

if nargin == 0
    nextstate = [0 0 0 0]';
    return
elseif nargin == 1
    nextstate = state;
    return
end

env = cart_environment();

switch action
    case 1 % Left
        F = -env.force;
    case 2 % Right
        F = env.force;
    otherwise
        error('Unknown action.')
end

x = state(1);
xd = state(2);
theta = state(3);
thetad = state(4);

costheta = cos(theta);
sintheta = sin(theta);

polemass_length = env.masspole * env.length;

F_effective = polemass_length * thetad^2 * sintheta + ...
    3 / 4 * env.masspole * costheta * (env.mu_p * thetad / polemass_length + env.g * sintheta);

masspole_effective = env.masspole * (1 - 3 / 4 * costheta^2);

xdd = F + F_effective - env.mu_c * sign(xd) / (env.masscart + masspole_effective);
thetadd = - 3 / 4 / env.length * (xdd * costheta + env.g * sintheta + env.mu_p * thetad / polemass_length);

x = x + env.dt .* xd;
xd = xd + env.dt .* xdd;
theta = theta + env.dt .* thetad;
theta = wrapinpi(theta); % theta in [-pi, pi]
thetad = thetad + env.dt .* thetadd;

nextstate = [x xd theta thetad]';

absorb = 0; % Infinite horizon
reward = 0;
% reward = cos(nexstate(3)); % Use with discount factor = 1
if max( (nextstate < env.minstate) | (nextstate > env.maxstate) )
    reward = -1;
    absorb = 1;
end

end