function [theta, ...     % parameters of the objective functions, theta = phi_rho(t)
    rho, ...             % parameters of theta
    t, ...               % free variable of theta
    D_t_theta, ...       % derivative of theta wrt t
    D_rho_theta, ...     % derivative of theta wrt rho
    J] = ...             % objective functions in closed form
    params_lqr(param_type)

mpd_var = lqr_mdpvariables();
dim_theta = mpd_var.nvar_reward;
    
if mpd_var.nvar_reward == 2
    
    if strcmp(param_type, 'P1') % Unconstrained
        dim_rho = 4;
        t = sym('t');
        rho = sym('rho',[1,dim_rho]);
        theta = sym('theta', [dim_theta,1]);
        theta(1) = sym('-1/(1+exp(rho1+rho2*t))');
        theta(2) = sym('-1/(1+exp(rho3+rho4*t))');
    elseif strcmp(param_type, 'P2') % Constrained
        dim_rho = 2;
        t = sym('t');
        rho = sym('rho',[1,dim_rho]);
        theta = sym('theta', [dim_theta,1]);
        theta(1) = sym('-1/(1+exp(-2.18708-rho1*t^2+(3.33837+rho1)*t))');
        theta(2) = sym('-1/(1+exp(1.15129-rho2*t^2+(-3.33837+rho2)*t))');
    else
        error('Unknown param type.');
    end

elseif mpd_var.nvar_reward == 3
    
    dim_rho = 9;
    t = sym('t',[1,2]);
    rho = sym('rho',[1,dim_rho]);
    theta = sym('theta', [dim_theta,1]);

    if strcmp(param_type, 'P1') % Unconstrained
        theta(1) = sym('-1/(1+exp(rho1 + rho2*t1 + rho3*t2))');
        theta(2) = sym('-1/(1+exp(rho4 + rho5*t1 + rho6*t2))');
        theta(3) = sym('-1/(1+exp(rho7 + rho8*t1 + rho9*t2))');
    elseif strcmp(param_type, 'P2') % Constrained
        theta(1) = sym('-1/(1+exp(1.151035476+rho1*t1-(-rho2+3.338299811)*t2-rho1*t1^2-rho2*t2^2-rho3*t2*t1))');
        theta(2) = sym('-1/(1+exp(1.151035476-(-rho4+3.338299811)*t1+rho5*t2-rho4*t1^2-rho5*t2^2-rho6*t1*t2))');
        theta(3) = sym('-1/(1+exp(-2.187264336-(-rho7-3.338299811)*t1-(-rho8-3.338299811)*t2-rho7*t1^2-rho8*t2^2-rho9*t1*t2))');
    else
        error('Unknown param type.');
    end
    
elseif mpd_var.nvar_reward == 5
    
    %%% Unconstrained
    t = sym('t',[1,4]);
    A = allcomb(t,t);
    A(13:15,:) = [];
    A(9:10,:) = [];
    A(5,:) = [];
    A = [t, prod(transpose(A))];

    rho_per_theta = length(A)+1;
    dim_rho = (rho_per_theta)*dim_theta;
    rho = sym('rho',[1,dim_rho]);
    theta = sym('theta', [dim_theta,1]);

    for i = 1 : dim_theta
        idx = rho_per_theta*(i-1);
        theta(i) = -1 / (1 + exp( sum([rho(1+idx), rho(1+idx+1:idx+rho_per_theta).*A]) ));
    end

end

D_t_theta = jacobian(theta,t);
D_rho_theta = jacobian(theta,rho);

if nargout == 6
    dim = mpd_var.nvar_reward;
    LQR   = lqr_init(dim);
    g     = LQR.g;
    B     = LQR.B;
    Q     = LQR.Q;
    R     = LQR.R;
    x0    = LQR.x0;
    Sigma = LQR.Sigma;
    
    J = sym('J',[1,dim]);
    K = diag(theta);
    
    for i = 1 : dim
        P = (Q{i}+K*R{i}*K)*(eye(dim)-g*(eye(dim)+2*K+K^2))^-1;
        J(i) = transpose(x0)*P*x0 + (1/(1-g))*trace(Sigma*(R{i}+g*transpose(B)*P*B));
    end
end