function K = kernel(X, Y, type, param)
% KERNEL Computes kernel function for matrices of points.
%
%    INPUT
%     - X     : N-by-D matrix (N samples, D dimension)
%     - Y     : M-by-D matrix (N samples, D dimension)
%     - type  : kernel name
%     - param : (optional) kernel parameters
%
%    OUTPUT
%     - K     : N-by-M kernel matrix

switch type
    
    case 'linear'
        K = X*Y';
        
    case 'poly'
        K = X*Y' + 1;
        if nargin < 4
            param = 2;
        end
        K = K.^param;
        
    case 'gauss'
        dist = L2_distance(X',Y');
        if nargin < 4
            tmp = dist;
            tmp(tmp==0) = inf;
            tmp = min(tmp);
            param = size(X,2)^2*mean(tmp);
        end
        K = exp(-dist.^2 ./ (2*param.^2));
        
    otherwise
        error('Unknown kernel.')
        
end

end
