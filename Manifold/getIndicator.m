function [L, D_J_L] = getIndicator(domain, ind_type, J, beta)
% Computes the quality of a solution according to a specific metric.
%
% Inputs:
% - domain   : the name of the MDP
% - ind_type : the type of indicator
% - J        : the solution to evaluate
% - beta     : parameters of the indicator function
%
% Outputs:
% - L        : the indicator
% - D_J_L    : the derivative of the indicator wrt J

[~, ~, utopia, antiutopia] = feval([domain '_moref'],0);
dim_J = length(J);

L_u = norm(J ./ utopia - ones(1,dim_J))^2;
L_au = norm(J ./ antiutopia - ones(1,dim_J))^2;
D_J_lau = 2 * (J - antiutopia) ./ antiutopia.^2;
D_J_lu = 2 * (J - utopia) ./ utopia.^2;

if strcmp(ind_type, 'utopia')
    L = -L_u;
    D_J_L = -D_J_lu;
elseif strcmp(ind_type, 'antiutopia')
    L = L_au;
    D_J_L = D_J_lau;
elseif strcmp(ind_type, 'mix2')
    L = beta(1) * L_au / L_u - beta(2);
    D_J_L = beta(1) * (D_J_lau * L_u - L_au * D_J_lu) / L_u^2;
elseif strcmp(ind_type, 'mix3')
    L = L_au * (1 - beta * L_u);
    D_J_L = D_J_lau * (1 - beta * L_u) - L_au * (beta * D_J_lu);
else
    error('Unknown loss function.')
end
