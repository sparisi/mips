function [J, ...         % objective functions
    theta, ...           % parameters of the objective functions, theta = phi_rho(t)
    r, ...               % parameters of theta
    t, ...               % free variable of theta
    D_theta_J, ...       % derivative of J wrt theta
    D2_theta_J, ...      % 2nd derivative of J wrt theta
    D_t_theta, ...       % derivative of theta wrt t
    D_r_theta, ...       % derivative of theta wrt rho
    I, ...               % indicator function
    I_params, ...        % indicator function parameters
    AUp, ...             % antiutopia point
    Up] = ...            % utopia point
    settings_lqr3( indicator_type, param_type )

dim   = 3;
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
t = sym('t',[1,2]);
theta = diag(K);

for i = 1 : dim
    P = (Q{i}+K*R{i}*K)*(eye(dim)-g*(eye(dim)+2*K+K^2))^-1;
    J(i) = transpose(x0)*P*x0 + (1/(1-g))*trace(Sigma*(R{i}+g*transpose(B)*P*B));
end

D_theta_J = jacobian(J,theta);
D2_theta_J = jacobian(D_theta_J(:),theta);

dim_r = 9;
r = sym('r',[1,dim_r]);
if strcmp(param_type, 'P1') % Unconstrained
    k1_1 = -1./(1+exp(r(1) + r(2)*t(1) + r(3)*t(2)));
    k2_2 = -1./(1+exp(r(4) + r(5)*t(1) + r(6)*t(2)));
    k3_3 = -1./(1+exp(r(7) + r(8)*t(1) + r(9)*t(2)));
elseif strcmp(param_type, 'P2') % Constrained
    k1_1 = -1./(1+exp(1.151035476+r(1)*t(1)-(-r(2)+3.338299811)*t(2)-r(1)*t(1)^2-r(2)*t(2)^2-r(3)*t(2)*t(1)));
    k2_2 = -1./(1+exp(1.151035476-(-r(4)+3.338299811)*t(1)+r(5)*t(2)-r(4)*t(1)^2-r(5)*t(2)^2-r(6)*t(1)*t(2)));
    k3_3 = -1./(1+exp(-2.187264336-(-r(7)-3.338299811)*t(1)-(-r(8)-3.338299811)*t(2)-r(7)*t(1)^2-r(8)*t(2)^2-r(9)*t(1)*t(2)));
else
    error('Unknown parameterization.');
end

K = subs(K);
theta = subs(theta);
D_t_theta = jacobian(theta,t);
J = subs(J);
D_theta_J = subs(D_theta_J);
D2_theta_J = subs(D2_theta_J);
D_r_theta = jacobian(theta,r);

[I, I_params] = parse_indicator(indicator_type, D_theta_J, J, Up, AUp);

end
