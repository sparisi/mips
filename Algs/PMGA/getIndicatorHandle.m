function [L, D_J_L] = getIndicatorHandle(ind_type, ind_params, utopia, antiutopia)
% GETINDICATORHANDLE Returns a handle to compute the desired 
% Pareto-optimality indicator of a set of solutions.
%
%    INPUT
%     - ind_type   : the type of indicator
%     - ind_params : parameters of the indicator function
%     - utopia     : utopia point, used for some indicators
%     - antiutopia : antiutopia point, used for some indicators
%
%    OUTPUT
%     - L          : the indicator handle, i.e., L = f(J), with J [N x D]
%                    matrix, where N is the number of points to evaluate 
%                    and D the number of objectives
%     - D_J_L      : the handle of the derivative of the indicator wrt J

L_u = @(J) norm(J ./ utopia - 1)^2;
L_au = @(J) norm(J ./ antiutopia - 1)^2;
D_J_lau = @(J) 2 * (J - antiutopia) ./ antiutopia.^2;
D_J_lu = @(J) 2 * (J - utopia) ./ utopia.^2;

switch ind_type
    case 'utopia'
        L = @(J) -L_u(J);
        D_J_L = @(J) -D_J_lu(J);
    case 'antiutopia'
        L = @(J) L_au(J);
        D_J_L = @(J) D_J_lau(J);
    case 'mix2'
        L = @(J) ind_params(1) * L_au(J) / L_u(J) - ind_params(2);
        D_J_L = @(J) ind_params(1) * (D_J_lau(J) * L_u(J) - L_au(J) * D_J_lu(J)) / L_u(J).^2;
    case 'mix3'
        L = @(J) L_au(J) * (1 - ind_params * L_u(J));
        D_J_L = @(J) D_J_lau(J) * (1 - ind_params * L_u(J)) - L_au(J) * (ind_params * D_J_lu(J));
    otherwise
        error('Unknown loss function.')
end
