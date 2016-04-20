function [I, D_J_I] = parse_indicator_handle(ind_type, ind_params, utopia, antiutopia)
% PARSE_INDICATOR_HANDLE Returns a handle to compute the desired 
% Pareto-optimality indicator of a set of solutions.
%
%    INPUT
%     - ind_type   : the type of indicator
%     - ind_params : parameters of the indicator function
%     - utopia     : utopia point, used for some indicators
%     - antiutopia : antiutopia point, used for some indicators
%
%    OUTPUT
%     - I          : the indicator handle, i.e., I = f(J), with J [N x D]
%                    matrix, where N is the number of points to evaluate 
%                    and D the number of objectives
%     - D_J_I      : the handle of the derivative of the indicator wrt J

I_u = @(J) norm(J ./ utopia - 1)^2;
I_au = @(J) norm(J ./ antiutopia - 1)^2;
D_J_Iau = @(J) 2 * (J - antiutopia) ./ antiutopia.^2;
D_J_Iu = @(J) 2 * (J - utopia) ./ utopia.^2;

switch ind_type
    case 'utopia'
        I = @(J) -I_u(J);
        D_J_I = @(J) -D_J_Iu(J);
    case 'antiutopia'
        I = @(J) I_au(J);
        D_J_I = @(J) D_J_Iau(J);
    case 'mix2'
        I = @(J) ind_params(1) * I_au(J) / I_u(J) - ind_params(2);
        D_J_I = @(J) ind_params(1) * (D_J_Iau(J) * I_u(J) - I_au(J) * D_J_Iu(J)) / I_u(J).^2;
    case 'mix3'
        I = @(J) I_au(J) * (1 - ind_params * I_u(J));
        D_J_I = @(J) D_J_Iau(J) * (1 - ind_params * I_u(J)) - I_au(J) * (ind_params * D_J_Iu(J));
    otherwise
        error('Unknown indicator.')
end
