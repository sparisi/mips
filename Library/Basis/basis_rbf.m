function Phi = basis_rbf(centers, widths, offset, state)
% BASIS_RBF Radial Basis Functions. 
% Phi = exp( -(state - centers)' * B * (state - centers) ), 
% where B is a diagonal matrix denoting the bandwiths of the kernels.
%
%    INPUT
%     - centers   : [D x C] matrix with C centers
%     - widths    : [D x 1] vector with the bandwidths (1/sigma^2)
%     - offset    : 1 to add an additional constant of value 1, 0 otherwise
%     - state     : (optional) [D x N] matrix of N states of size D to 
%                   evaluate
%
%    OUTPUT
%     - Phi       : if a state is provided as input, the function 
%                   returns the feature vectors representing it; 
%                   otherwise it returns the number of features

if nargin < 4
    Phi = size(centers,2) + 1*(offset == 1);
else
    stateB = bsxfun(@times, state, sqrt(widths));
    centersB = bsxfun(@times, centers, sqrt(widths));
    stateB = permute(stateB,[3,2,1]);
    centersB = permute(centersB,[2,3,1]);
    Phi = exp(sum(-(centersB-stateB).^2,3));

%     Phi = bsxfun(@times, Phi, 1 ./ sum(Phi,1));
    if offset == 1, Phi = [ones(1,size(state,2)); Phi]; end
end
