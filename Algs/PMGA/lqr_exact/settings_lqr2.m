function [J, ...         % objective functions
    theta, ...           % parameters of the objective functions, theta = phi_rho(t)
    r, ...               % parameters of theta
    t, ...               % free variable of theta
    D_theta_J, ...       % derivative of J wrt theta
    D2_theta_J, ...      % 2nd derivative of J wrt theta
    D_t_theta, ...       % derivative of theta wrt t
    D_r_theta, ...       % derivative of theta wrt rho
    L, ...               % loss function
    L_params, ...        % loss function parameters
    AUp, ...             % antiutopia point
    Up] = ...            % utopia point
    settings_lqr2( loss_type, param_type )

%% Initialization
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
else
    error('Unknown param type.');
end

K = subs(K);
r = sym('r',[1,dim_r]);
theta = subs(theta);
D_t_theta = jacobian(theta,t);
J = subs(J);
D_theta_J = subs(D_theta_J);
D2_theta_J = subs(D2_theta_J);
D_r_theta = jacobian(theta,r);


%% Loss functions
L_params = {};
if strcmp('pareto',loss_type)
    L = -paretoDirectionNorm(D_theta_J);
elseif strcmp('utopia',loss_type)
    L = -norm(J./Up-ones(size(J)))^2;
elseif strcmp('antiutopia',loss_type)
    L = norm(J./AUp-ones(size(J)))^2;
elseif strcmp('sum_J',loss_type)
    L = sum(J);
elseif strcmp('mix1',loss_type)
    L_params = sym('beta');
    L_antiutopia = norm(J./AUp-ones(size(J)))^2;
    L_pa = paretoDirectionNorm(D_theta_J);
    L = L_antiutopia*(1-L_params*L_pa);
elseif strcmp('mix2',loss_type)
    L_params = sym('beta',[1,2]);
    L_utopia = norm(J./Up-ones(size(J)))^2;
    L_antiutopia = norm(J./AUp-ones(size(J)))^2;
    L = L_params(1) * L_antiutopia / L_utopia - L_params(2);
elseif strcmp('mix3',loss_type)
    L_params = sym('beta');
    L_antiutopia = norm(J./AUp-ones(size(J)))^2;
    L_utopia = norm(J./Up-ones(size(J)))^2;
    L = L_antiutopia*(1-L_params*L_utopia);
else
    error('Unknown loss function.')
end

end
