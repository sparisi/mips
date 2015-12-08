function Phi = basis_krbf(n_centers, range, state)
% BASIS_KRBF Uniformly distributed Kernel Radial Basis Functions. Centers  
% and bandwidths are automatically computed to guarantee 25% of overlapping  
% and peaks between 0.95-0.99.
% Phi = exp( -0.5 * (state - centers)' * B^-1 * (state - centers) ), 
% where B is a diagonal matrix denoting the bandwiths of the kernels.
%
%    INPUT
%     - n_centers : number of centers (the same for all dimensions)
%     - range     : N-by-2 matrix with min and max values for the
%                   N-dimensional input state
%     - state     : (optional) X-by-Y matrix with the Y states of size X 
%                   to evaluate
%
%    OUTPUT
%     - Phi       : if a state is provided as input, the function 
%                   returns the feature vectors representing it; 
%                   otherwise it returns the number of features
%
% =========================================================================
% EXAMPLE
% basis_krbf(2, [0,1; 0,1], [0.2, 0.1]')
%     0.7118
%     0.0508
%     0.0211
%     0.0015

persistent centers bands

% Compute bandwidths and centers only once
if isempty(centers)
    n_features = size(range,1);
    bands = diff(range,[],2).^2 / n_centers^3;
    m = diff(range,[],2) / n_centers;
    
    c = cell(n_features, 1);
    for i = 1 : n_features
        c{i} = linspace(-m(i) * 0.1 + range(i,1), range(i,2) + m(i) * 0.1, n_centers);
    end
    d = cell(1,n_features);
    [d{:}] = ndgrid(c{:});
    centers = cell2mat( cellfun(@(v)v(:), d, 'UniformOutput', false) )';
end

if nargin < 3
    Phi = size(centers,2);
else
    B = 0.5 ./ bands;
    distance = -bsxfun(@minus,state',reshape(centers,[1 size(centers)])).^2;
    distance = permute(distance,[1 3 2]);
    expterm = bsxfun(@times,distance,reshape(B',[1,size(B')]));
    Phi = exp(sum(expterm,3))';
%     Phi = Phi ./ sum(Phi);
end
