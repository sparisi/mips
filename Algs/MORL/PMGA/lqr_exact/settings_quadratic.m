function [J, ...         % objective functions
    theta, ...           % parameters of the objective functions, theta = phi_rho(t)
    r, ...               % parameters of theta
    t, ...               % free variable of theta
    D_theta_J, ...       % derivative of J wrt theta
    D2_theta_J, ...      % 2nd derivative of J wrt theta
    D_t_theta, ...       % derivative of theta wrt t
    D_r_theta, ...       % derivative of theta wrt rho
    I, ...               % loss function
    I_params, ...        % loss function parameters
    AUp, ...             % antiutopia point
    Up] = ...            % utopia point
    settings_quadratic( indicator_type )

% Quadratic reward (theta - goal).^2
% We use different goals for each objective (e.g., 1 and -1).

dim = 2; % Change the dimensionality of the problem

Up = sym('Up',[1,dim]);
AUp = sym('AUp',[1,dim]);

% t = sym('t',[1,3]);
t = sym('t');
theta = sym('theta',[dim,1]);
J = [mean(theta-1).^2, mean(theta+1).^2];

D_theta_J = jacobian(J,theta);
D2_theta_J = jacobian(D_theta_J(:),theta);

% Neural network manifold
dim_theta = length(theta);
dim_t = length(t);
d1 = 3; % Hidden layer size
dim_r = dim_t*d1+d1+d1*dim_theta+dim_theta;
r = sym('r',[1,dim_r]);

% One hidden layer with tanh + sigmoid at the end to bound theta in [-1,0]
net = transpose( ...
    - 1 ./ (1 + exp( ...
    - (tanh( ...
    t*reshape(r(1:dim_t*d1),[dim_t,d1])+r(dim_t*d1+1:dim_t*d1+d1))*reshape(r(dim_t*d1+d1+1:dim_t*d1+d1+dim_theta*d1),d1,dim_theta)+r(end-dim_theta+1:end))) ));

% Do not do theta(1) = net(1) or the subs(theta) below won't work
theta1 = net(1);
theta2 = net(2);

theta = subs(theta);
D_t_theta = jacobian(theta,t);
J = subs(J);
D_theta_J = subs(D_theta_J);
D2_theta_J = subs(D2_theta_J);
D_r_theta = jacobian(theta,r);

[I, I_params] = parse_indicator(indicator_type, D_theta_J, J, Up, AUp);

end
