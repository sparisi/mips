function K = kernel(X, Y, type, param)
% KERNEL Computes kernel function for matrices of points.
%
%    INPUT
%     - X     : [N x D] matrix (N samples, D dimension)
%     - Y     : [M x D] matrix (M samples, D dimension)
%     - type  : kernel name
%     - param : kernel parameters
%
%    OUTPUT
%     - K     : [N x M] kernel matrix

switch type
    
    case 'linear'
        K = X*Y';
        
    case 'poly'
        K = X*Y' + 1;
        K = K.^param;
        
    case 'gauss'
        dist_sq = L2_distance_sq(X',Y');
        K = exp(-dist_sq ./ (2*param.^2));
        
    case 'epanechnikov'
        dist_sq = L2_distance_sq(X',Y');
        K = max(0, 1 - exp(-dist_sq ./ (2*param.^2)));
        
    otherwise
        error('Unknown kernel.')
        
end

end
