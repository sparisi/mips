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
    settings_lqr2( indicator_type, param_type )

dim   = 2;
LQR   = lqr_init(dim);
g     = LQR.g;
A     = LQR.A;
B     = LQR.B;
Q     = LQR.Q;
R     = LQR.R;
x0    = LQR.x0;
Sigma = LQR.Sigma;

Up = sym('Up',[1,dim]);
AUp = sym('AUp',[1,dim]);
    
J = sym('J',[1,dim]);
K = sym('k',[dim,dim]);
K(~eye(dim)) = 0;
t = sym('t');
theta = diag(K);

for i = 1 : dim
    P = (Q{i}+K*R{i}*K)*(eye(dim)-g*(eye(dim)+2*K+K^2))^-1;
    J(i) = transpose(x0)*P*x0 + (1/(1-g))*trace(Sigma*(R{i}+g*transpose(B)*P*B));
end

D_theta_J = jacobian(J,theta);
D2_theta_J = jacobian(D_theta_J(:),theta);

if strcmp(param_type, 'P1') % Unconstrained
    k1_1 = sym('-1/(1+exp(r1+r2*t))');
    k2_2 = sym('-1/(1+exp(r3+r4*t))');
    dim_r = 4;
elseif strcmp(param_type, 'P2') % Constrained
    k1_1 = sym('-1/(1+exp(-2.18708-r1*t^2+(3.33837+r1)*t))');
    k2_2 = sym('-1/(1+exp(1.15129-r2*t^2+(-3.33837+r2)*t))');
    dim_r = 2;
elseif strcmp(param_type, 'NN') % Neural network (unconstrained)
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
    k1_1 = net(1);
    k2_2 = net(2);
else
    error('Unknown parameterization.');
end

K = subs(K);
r = sym('r',[1,dim_r]);
theta = subs(theta);
D_t_theta = jacobian(theta,t);
J = subs(J);
D_theta_J = subs(D_theta_J);
D2_theta_J = subs(D2_theta_J);
D_r_theta = jacobian(theta,r);

[I, I_params] = parse_indicator(indicator_type, D_theta_J, J, Up, AUp);

end
