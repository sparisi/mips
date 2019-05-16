function [theta, ...     % parameters of the objective functions, theta = phi_rho(t)
    rho, ...             % parameters of theta
    t, ...               % free variable of theta
    D_t_theta, ...       % derivative of theta wrt t
    D_rho_theta] = ...   % derivative of theta wrt rho
    params_dam (param_type, num_obj)

dim_theta = 6;
dim_rho = 6;
t = sym('t',[1,num_obj-1]);
rho = sym('rho',[1,dim_rho]);
theta = sym('theta', [dim_theta,1]);

if num_obj == 2
    
    theta(1) = str2sym('66 - rho1*t1^2 + (-16 + rho1)*t1');
    theta(2) = str2sym('-105 - rho2*t1^2 + (20 + rho2)*t1');
    theta(3) = str2sym('18 - rho3*t1^2 + (-16 + rho3)*t1');
    theta(4) = str2sym('-23 - rho4*t1^2 + (53 + rho4)*t1');
    theta(5) = str2sym('39 - rho5*t1^2 + (121 + rho5)*t1');
    theta(6) = str2sym('0.01 - rho6*t1^2 + (0.1 + rho6)*t1');
  
else
    
    theta(1) = str2sym('36 + 2*(-0.5*rho1+7.5)*t2 + (rho1+1)*t1*t2 + 30*t1^2 + (rho1-1)*t2^2');
    theta(2) = str2sym('-57 + (2*(-0.5*rho2-13.5))*t2 + (rho2+1)*t1*t2 - 48*t1^2 + (rho2-1)*t2^2');
    theta(3) = str2sym('13 + (-2*rho3+7)*t1 + (rho3+1)*t1*t2 + (2*(rho3-1))*t1^2 - 11*t2^2');
    theta(4) = str2sym('-30 + (-2*rho4+9)*t1 + (rho4+1)*t1*t2 + (2*(rho4-1))*t1^2 + 60*t2^2');
    theta(5) = str2sym('104 + (2*(-0.5*rho5+28.5))*t2 + (rho5+1)*t1*t2 - 65*t1^2 + (rho5-1)*t2^2');
    theta(6) = str2sym('0.05 + (2*(-0.5*rho6+0.5))*t2 + (rho6+1)*t1*t2 + (rho6-1)*t2^2');

end
    
D_t_theta = jacobian(theta,t);
D_rho_theta = jacobian(theta,rho);

end
