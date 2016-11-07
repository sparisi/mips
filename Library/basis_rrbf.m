function Phi = basis_rrbf(n_centers, widths, range, offset, state)
% BASIS_RRBF Uniformly distributed Roooted Gaussian Radial Basis Functions.
% Phi(i) = exp(-||state - centers(i)|| / widths(i))
%
%    INPUT
%     - n_centers : number of centers (the same for all dimensions)
%     - widths    : array of widths for each dimension
%     - range     : [D x 2] matrix with min and max values for the
%                   D-dimensional input state
%     - offset    : 1 to add an additional constant of value 1, 0 otherwise
%     - state     : (optional) [D x N] matrix of N states of size D to 
%                   evaluate
%
%    OUTPUT
%     - Phi       : if a state is provided as input, the function 
%                   returns the feature vectors representing it; 
%                   otherwise it returns the number of features
%
% =========================================================================
% EXAMPLE
% basis_rrbf(2, [0.3; 0.2], [0 1; 0 1], 0, [0.2; 0.1])
%     0.4346
%     0.0663
%     0.0106
%     0.0053

persistent centers

dim_state = size(range,1);

% Compute all centers point only once
if isempty(centers)
    c = cell(dim_state, 1);
    for i = 1 : dim_state
        c{i} = linspace(range(i,1), range(i,2), n_centers);
    end
    
    d = cell(1,dim_state);
    [d{:}] = ndgrid(c{:});
    centers = cell2mat( cellfun(@(v)v(:), d, 'UniformOutput',false) )';
end

if nargin < 5
    Phi = size(centers,2) + 1*(offset == 1);
else
    if isrow(widths), widths = widths'; end
    B = 1./widths.^2;
    distance = bsxfun(@minus,state',reshape(centers,[1 size(centers)])).^2;
    distance = permute(distance,[1 3 2]);
    expterm = bsxfun(@times,distance,reshape(B',[1,size(B')]));
    Phi = exp(-sqrt(sum(expterm,3)))';
%     Phi = bsxfun(@times, Phi, 1 ./ sum(Phi,1));
    if offset == 1, Phi = [ones(1,size(state,2)); Phi]; end
end

