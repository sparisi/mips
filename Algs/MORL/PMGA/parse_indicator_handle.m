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

I_u = @(J) sum((bsxfun(@times, J, 1 ./ utopia) - 1).^2, 2);
I_au = @(J) sum((bsxfun(@times, J, 1 ./ antiutopia) - 1).^2, 2);
D_J_Iau = @(J) 2 * bsxfun(@times, bsxfun(@minus, J, antiutopia), 1 ./ antiutopia.^2);
D_J_Iu = @(J) 2 * bsxfun(@times, bsxfun(@minus, J, utopia), 1 ./ utopia.^2);

switch ind_type
    case 'utopia'
        I = @(J,varargin) -I_u(J);
        D_J_I = @(J,varargin) -D_J_Iu(J);
    case 'antiutopia'
        I = @(J,varargin) I_au(J);
        D_J_I = @(J,varargin) D_J_Iau(J);
    case 'mix2'
        I = @(J,varargin) ind_params(1) .* I_au(J) ./ I_u(J) - ind_params(2);
        D_J_I = @(J,varargin) ind_params(1) .* (D_J_Iau(J) .* I_u(J) - I_au(J) .* D_J_Iu(J)) ./ I_u(J).^2;
    case 'mix3'
        I = @(J,varargin) I_au(J) .* (1 - ind_params .* I_u(J));
        D_J_I = @(J) D_J_Iau(J) .* (1 - ind_params .* I_u(J)) - I_au(J) .* (ind_params .* D_J_Iu(J));
    case 'hv'
        P = bsxfun( @plus, antiutopia, bsxfun(@times, (utopia - antiutopia), rand(1e1, length(utopia))) );
        I = @(J,varargin) metric_hv_relu(J,P,ind_params(1),ind_params(2));
        D_J_I = @(J,varargin) metric_hv_relu_d(J,P,ind_params(1),ind_params(2));
    otherwise
        error('Unknown indicator.')
end

end
