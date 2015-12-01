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
        force = -env.force_mag;
    case 2 % Right
        force = env.force_mag;
    otherwise
        error('Unknown action.')
end

x = state(1);
xd = state(2);
theta = state(3);
thetad = state(4);

costheta = cos(theta);
sintheta = sin(theta);

totalmass = env.masspole + env.masscart;
polemass_length = env.masspole * env.length;

temp = (force + polemass_length * thetad^2 * sintheta) / totalmass;
thetadd = (env.g * sintheta - costheta * temp) / (env.length * (4/3 - env.masspole * costheta^2 / totalmass));
xdd = temp - polemass_length * thetadd * costheta / totalmass;

x = x + env.dt * xd;
xd = xd + env.dt * xdd;
theta = theta + env.dt * thetad;
theta = wrapinpi(theta); % theta in [-pi, pi]
thetad = thetad + env.dt * thetadd;

nextstate = [x xd theta thetad]';

absorb = 0; % Infinite horizon
reward = 0;
% reward = cos(nexstate(3)); % Use with discount factor = 1
if max( (nextstate < env.minstate) | (nextstate > env.maxstate) )
    reward = -1;
    absorb = 1;
end

end