function Phi = basis_rrbf(n_centers, widths, range, state)
% BASIS_RRBF Uniformly distributed Roooted Gaussian Radial Basis Functions.
% Phi(i) = exp(-||state - centers(i)|| / widths(i))
%
%    INPUT
%     - n_centers : number of centers (the same for all dimensions)
%     - widths    : array of widths for each dimension
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
% basis_rrbf(2, [30, 20], [0,1; 0,1], [0.2, 0.1]')
%     0.2983
%     0.0499
%     0.0007
%     0.0000

persistent centers

n_features = size(range,1);
c = cell(n_features, 1);

% Compute centers for each dimension
for i = 1 : n_features
    c{i} = linspace(range(i,1), range(i,2), n_centers);
end

% Compute all centers point
if size(centers,1) == 0
    d = cell(1,n_features);
    [d{:}] = ndgrid(c{:});
    centers = cell2mat( cellfun(@(v)v(:), d, 'UniformOutput',false) )';
end
dim_phi = size(centers,2);

if ~exist('state','var')
    
    Phi = dim_phi;
    
else

    B = diag(1./widths.^2);
    N = size(state,2);
    Phi = zeros(dim_phi,N);
    for j = 1 : N
        x = bsxfun(@minus,state(:,j),centers);
        Phi(:,j) = diag(exp( -sqrt( x' * B * x ) ));
    end
    
end

% %%% Plotting
% idx = 1;
% t = zeros(100, n_features);
% for i = 1 : n_features
%     t(:,i)  = linspace(range(i,1), range(i,2), 100);
% end
% 
% u = zeros(100, n_centers);
% for k = 1 : 100
%     for j = 1 : n_centers
%         u(k,j) = exp(-(t(k,idx) - c{idx}(j))^2 / (2*b(idx)));
%     end
%     u(k,:) = u(k,:) / sum(u(k,:));
% end
% 
% figure; plot(t(:,idx),u,'Linewidth',2)

end
