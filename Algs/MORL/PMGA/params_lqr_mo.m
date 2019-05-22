function [theta, ...     % parameters of the objective functions, theta = phi_rho(t)
    rho, ...             % parameters of theta
    t, ...               % free variable of theta
    D_t_theta, ...       % derivative of theta wrt t
    D_rho_theta, ...     % derivative of theta wrt rho
    J] = ...             % objective functions in closed form
    params_lqr_mo(param_type, num_obj)

dim_theta = num_obj;
dim_t = dim_theta-1;
t = sym('t',[1,dim_t]);
theta = sym('theta', [dim_theta,1]);


if strcmp(param_type, 'NN') % Neural network (unconstrained) for any dimension

    d1 = 16; % Hidden layer size
    dim_rho = dim_t*d1+d1+d1*dim_theta+dim_theta;
    rho = sym('rho',[1,dim_rho]);
    % One hidden layer with tanh + sigmoid at the end to bound theta in [-1,0]
    theta = transpose( ...
        - 1 ./ (1 + exp( ...
        - (tanh( ...
        t*reshape(rho(1:dim_t*d1),[dim_t,d1])+rho(dim_t*d1+1:dim_t*d1+d1))*reshape(rho(dim_t*d1+d1+1:dim_t*d1+d1+dim_theta*d1),d1,dim_theta)+rho(end-dim_theta+1:end))) ));

elseif dim_theta == 2
    
    if strcmp(param_type, 'P1') % Unconstrained
        dim_rho = 4;
        rho = sym('rho',[1,dim_rho]);
        theta(1) = -1./(1+exp(rho(1)+rho(2)*t(1)));
        theta(2) = -1./(1+exp(rho(3)+rho(4)*t(1)));
    
    elseif strcmp(param_type, 'P2') % Constrained
        dim_rho = 2;
        rho = sym('rho',[1,dim_rho]);
        theta = sym('theta', [dim_theta,1]);
        theta(1) = -1./(1+exp(-2.18708-rho(1)*t(1)^2+(3.33837+rho(1))*t(1)));
        theta(2) = -1./(1+exp(1.15129-rho(2)*t(1)^2+(-3.33837+rho(2))*t(1)));
    else
        error('Unknown param type.');
    end

elseif dim_theta == 3
    
    dim_rho = 9;
    rho = sym('rho',[1,dim_rho]);

    if strcmp(param_type, 'P1') % Unconstrained
        theta(1) = -1./(1+exp(rho(1) + rho(2)*t(1) + rho(3)*t(2)));
        theta(2) = -1./(1+exp(rho(4) + rho(5)*t(1) + rho(6)*t(2)));
        theta(3) = -1./(1+exp(rho(7) + rho(8)*t(1) + rho(9)*t(2)));
    elseif strcmp(param_type, 'P2') % Constrained
        theta(1) = -1./(1+exp(1.151035476+rho(1)*t(1)-(-rho(2)+3.338299811)*t(2)-rho(1)*t(1)^2-rho(2)*t(2)^2-rho(3)*t(2)*t(1)));
        theta(2) = -1./(1+exp(1.151035476-(-rho(4)+3.338299811)*t(1)+rho(5)*t(2)-rho(4)*t(1)^2-rho(5)*t(2)^2-rho(6)*t(1)*t(2)));
        theta(3) = -1./(1+exp(-2.187264336-(-rho(7)-3.338299811)*t(1)-(-rho(8)-3.338299811)*t(2)-rho(7)*t(1)^2-rho(8)*t(2)^2-rho(9)*t(1)*t(2)));
    else
        error('Unknown param type.');
    end
    
elseif dim_theta == 5
    
    %%% Unconstrained
    A = allcomb(t,t);
    A(13:15,:) = [];
    A(9:10,:) = [];
    A(5,:) = [];
    A = [t, prod(transpose(A))];

    rho_per_theta = length(A)+1;
    dim_rho = (rho_per_theta)*dim_theta;
    rho = sym('rho',[1,dim_rho]);

    for i = 1 : dim_theta
        idx = rho_per_theta*(i-1);
        theta(i) = -1 / (1 + exp( sum([rho(1+idx), rho(1+idx+1:idx+rho_per_theta).*A]) ));
    end

end

D_t_theta = jacobian(theta,t);
D_rho_theta = jacobian(theta,rho);

if nargout == 6
    dim   = num_obj;
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