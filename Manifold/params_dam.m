function [theta, ...     % parameters of the objective functions, theta = phi_rho(t)
    rho, ...             % parameters of theta
    t, ...               % free variable of theta
    D_t_theta, ...       % derivative of theta wrt t
    D_rho_theta] = ...   % derivative of theta wrt rho
    params_dam ()

mpd_var = dam_mdpvariables();

if mpd_var.nvar_reward == 2
    
    dim_theta = 6;
    dim_rho = 6;
    t = sym('t');
    rho = sym('rho',[1,dim_rho]);
    theta = sym('theta', [dim_theta,1]);

    %%% Squared RBF
%     theta(1) = sym('95.5718 - rho1*t^2 + (-50.0692 + rho1)*t');
%     theta(2) = sym('-80.5557 - rho2*t^2 + (62.5151 + rho2)*t');
%     theta(3) = sym('-27.8499 - rho3*t^2 + (32.6664 + rho3)*t');
%     theta(4) = sym('21.7073 - rho4*t^2 + (-17.2786 + rho4)*t');
%     theta(5) = sym('132.2700 - rho5*t^2 + (-127.9720 + rho5)*t');
%     theta(6) = sym('0.01 - rho6*t^2 + (0.1 + rho6)*t');
    
    %%% Non-squared RBF
    theta(1) = sym('66 - rho1*t^2 + (-16 + rho1)*t');
    theta(2) = sym('-105 - rho2*t^2 + (20 + rho2)*t');
    theta(3) = sym('18 - rho3*t^2 + (-16 + rho3)*t');
    theta(4) = sym('-23 - rho4*t^2 + (53 + rho4)*t');
    theta(5) = sym('39 - rho5*t^2 + (121 + rho5)*t');
    theta(6) = sym('0.01 - rho6*t^2 + (0.1 + rho6)*t');
  
else
    
    dim_theta = 6;
    dim_rho = 6;
    t = sym('t',[1,2]);
    rho = sym('rho',[1,dim_rho]);
    theta = sym('theta', [dim_theta,1]);

    %%% Squared RBF
%     theta(1) = sym('95.5718 + (-2*rho1-48.0692)*t1 + (rho1+1)*t1*t2 + (2*(rho1-1))*t1^2 - 54.448*t2^2');
%     theta(2) = sym('-80.5557 + (2*(-0.5*rho2+18.34715))*t2 + (rho2+1)*t1*t2 + 62.5151*t1^2 + (rho2-1)*t2^2');
%     theta(3) = sym('-27.8499 + (2*(-0.5*rho3+9.4146))*t2 + (rho3+1)*t1*t2 + 32.6664*t1^2 + (rho3-1)*t2^2');
%     theta(4) = sym('21.7073 + (2*(-0.5*rho4-12.84175))*t2 + (rho4+1)*t1*t2 - 17.2786*t1^2 + (rho4-1)*t2^2');
%     theta(5) = sym('132.27 + (2*(-0.5*rho5-40.62905))*t2 + (rho5+1)*t1*t2 - 127.972*t1^2 + (rho5-1)*t2^2');
%     theta(6) = sym('0.1+(2*(-0.5*rho6+0.5))*t2 + (rho6+1)*t1*t2 + (rho6-1)*t2^2');

    %%% Non-squared RBF
    theta(1) = sym('36 + 2*(-0.5*rho1+7.5)*t2 + (rho1+1)*t1*t2 + 30*t1^2 + (rho1-1)*t2^2');
    theta(2) = sym('-57 + (2*(-0.5*rho2-13.5))*t2 + (rho2+1)*t1*t2 - 48*t1^2 + (rho2-1)*t2^2');
    theta(3) = sym('13 + (-2*rho3+7)*t1 + (rho3+1)*t1*t2 + (2*(rho3-1))*t1^2 - 11*t2^2');
    theta(4) = sym('-30 + (-2*rho4+9)*t1 + (rho4+1)*t1*t2 + (2*(rho4-1))*t1^2 + 60*t2^2');
    theta(5) = sym('104 + (2*(-0.5*rho5+28.5))*t2 + (rho5+1)*t1*t2 - 65*t1^2 + (rho5-1)*t2^2');
    theta(6) = sym('0.05 + (2*(-0.5*rho6+0.5))*t2 + (rho6+1)*t1*t2 + (rho6-1)*t2^2');

end
    
D_t_theta = jacobian(theta,t);
D_rho_theta = jacobian(theta,rho);

end
