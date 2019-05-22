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
    
    theta(1) = 66 - rho(1)*t(1)^2 + (-16 + rho(1))*t(1);
    theta(2) = -105 - rho(2)*t(1)^2 + (20 + rho(2))*t(1);
    theta(3) = 18 - rho(3)*t(1)^2 + (-16 + rho(3))*t(1);
    theta(4) = -23 - rho(4)*t(1)^2 + (53 + rho(4))*t(1);
    theta(5) = 39 - rho(5)*t(1)^2 + (121 + rho(5))*t(1);
    theta(6) = 0.01 - rho(6)*t(1)^2 + (0.1 + rho(6))*t(1);
  
else
    
    theta(1) = 36 + 2*(-0.5*rho(1)+7.5)*t(2) + (rho(1)+1)*t(1)*t(2) + 30*t(1)^2 + (rho(1)-1)*t(2)^2;
    theta(2) = -57 + (2*(-0.5*rho(2)-13.5))*t(2) + (rho(2)+1)*t(1)*t(2) - 48*t(1)^2 + (rho(2)-1)*t(2)^2;
    theta(3) = 13 + (-2*rho(3)+7)*t(1) + (rho(3)+1)*t(1)*t(2) + (2*(rho(3)-1))*t(1)^2 - 11*t(2)^2;
    theta(4) = -30 + (-2*rho(4)+9)*t(1) + (rho(4)+1)*t(1)*t(2) + (2*(rho(4)-1))*t(1)^2 + 60*t(2)^2;
    theta(5) = 104 + (2*(-0.5*rho(5)+28.5))*t(2) + (rho(5)+1)*t(1)*t(2) - 65*t(1)^2 + (rho(5)-1)*t(2)^2;
    theta(6) = 0.05 + (2*(-0.5*rho(6)+0.5))*t(2) + (rho(6)+1)*t(1)*t(2) + (rho(6)-1)*t(2)^2;

end
    
D_t_theta = jacobian(theta,t);
D_rho_theta = jacobian(theta,rho);

end
