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

if strcmp(param_type, 'P1') % Unconstrained
    k1_1 = sym('-1/(1+exp(r1 + r2*t1 + r3*t2))');
    k2_2 = sym('-1/(1+exp(r4 + r5*t1 + r6*t2))');
    k3_3 = sym('-1/(1+exp(r7 + r8*t1 + r9*t2))');
    dim_r = 9;
elseif strcmp(param_type, 'P2') % Constrained
    k1_1 = sym('-1/(1+exp(1.151035476+r1*t1-(-r2+3.338299811)*t2-r1*t1^2-r2*t2^2-r3*t2*t1))');
    k2_2 = sym('-1/(1+exp(1.151035476-(-r4+3.338299811)*t1+r5*t2-r4*t1^2-r5*t2^2-r6*t1*t2))');
    k3_3 = sym('-1/(1+exp(-2.187264336-(-r7-3.338299811)*t1-(-r8-3.338299811)*t2-r7*t1^2-r8*t2^2-r9*t1*t2))');
    dim_r = 9;
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
