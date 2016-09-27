function Phi = basis_krbf(n_centers, range, offset, state)
% BASIS_KRBF Kernel Radial Basis Functions. 
% Phi = exp( -0.5 * (state - centers)' * B^-1 * (state - centers) ), 
% where B is a diagonal matrix denoting the bandwiths of the kernels.
% Centers are uniformly placed in RANGE and bandwidths are automatically 
% computed. See the code for more details.
%
%    INPUT
%     - n_centers : number of centers (the same for all dimensions)
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

persistent centers bands

% Compute bandwidths and centers only once
if isempty(centers)
    dim_state = size(range,1);
    bands = diff(range,[],2).^2 / (2 * n_centers^2); % change the denominator for wider/narrower bandwidth
    m = diff(range,[],2) / n_centers;
    
    c = cell(dim_state, 1);
    for i = 1 : dim_state
        c{i} = linspace(-m(i) * 0.1 + range(i,1), range(i,2) + m(i) * 0.1, n_centers);
    end
    d = cell(1,dim_state);
    [d{:}] = ndgrid(c{:});
    centers = cell2mat( cellfun(@(v)v(:), d, 'UniformOutput', false) )';
end

if nargin < 4
    Phi = size(centers,2) + 1*(offset == 1);
else
    B = 0.5 ./ bands;
    distance = -bsxfun(@minus,state',reshape(centers,[1 size(centers)])).^2;
    distance = permute(distance,[1 3 2]);
    expterm = bsxfun(@times,distance,reshape(B',[1,size(B')]));
    Phi = exp(sum(expterm,3))';
%     Phi = Phi ./ sum(Phi);
    if offset == 1, Phi = [ones(1,size(state,2)); Phi]; end
end
