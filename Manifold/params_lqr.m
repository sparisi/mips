function [theta, ...     % parameters of the objective functions, theta = phi_rho(t)
    rho, ...             % parameters of theta
    t, ...               % free variable of theta
    D_t_theta, ...       % derivative of theta wrt t
    D_rho_theta] = ...   % derivative of theta wrt rho
    params_lqr( )

mpd_var = dam_mdpvariables();
    
if mpd_var.nvar_reward == 2
    
    dim_theta = 2;

    %%% Unconstrained
    dim_rho = 4;
    t = sym('t');
    rho = sym('rho',[1,dim_rho]);
    theta = sym('theta', [dim_theta,1]);

    theta(1) = sym('-1/(1+exp(rho1+rho2*t))');
    theta(2) = sym('-1/(1+exp(rho3+rho4*t))');

    %%% Constrained
%     dim_rho = 2;
%     t = sym('t');
%     rho = sym('rho',[1,dim_rho]);
%     theta = sym('theta', [dim_theta,1]);
%     theta(1) = sym('-1/(1+exp(-2.18708-rho1*t^2+(3.33837+rho1)*t))');
%     theta(2) = sym('-1/(1+exp(1.15129-rho2*t^2+(-3.33837+rho2)*t))');

else
    
    dim_theta = 3;
    dim_rho = 9;
    t = sym('t',[1,2]);
    rho = sym('rho',[1,dim_rho]);
    theta = sym('theta', [dim_theta,1]);

    %%% Constrained
    theta(1) = sym('-1/(1+exp(1.151035476+rho1*t1-(-rho2+3.338299811)*t2-rho1*t1^2-rho2*t2^2-rho3*t2*t1))');
    theta(2) = sym('-1/(1+exp(1.151035476-(-rho4+3.338299811)*t1+rho5*t2-rho4*t1^2-rho5*t2^2-rho6*t1*t2))');
    theta(3) = sym('-1/(1+exp(-2.187264336-(-rho7-3.338299811)*t1-(-rho8-3.338299811)*t2-rho7*t1^2-rho8*t2^2-rho9*t1*t2))');

    %%% Unconstrained
%     theta(1) = sym('-1/(1+exp(rho1 + rho2*t1 + rho3*t2))');
%     theta(2) = sym('-1/(1+exp(rho4 + rho5*t1 + rho6*t2))');
%     theta(3) = sym('-1/(1+exp(rho7 + rho8*t1 + rho9*t2))');
    
end

D_t_theta = jacobian(theta,t);
D_rho_theta = jacobian(theta,rho);
