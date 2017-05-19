function Phi = basis_tiles(n_centers, range, offset, state)
% BASIS_TILES Tile coding basis functions. 
% The state space is divided into a grid, with centers uniformly placed in 
% RANGE. For each state, only the tile with the closest center is active.
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
%
% =========================================================================
% EXAMPLE
% basis_tiles(5, [[1,5];[1,5]], 0, [2.2; 4.1])
% that is, we have 2D states in [1,5]x[1,5] divided in 5x5 bins. 
% If we input state [2.2; 4.1], then only the center at [2,4] is active. 
% The feature vector is the vectorized matrix below with all 0 but a 1.
% 
%    - - - - -
% 5 |0|0|0|0|0|
%    - - - - -
% 4 |0|1|0|0|0|
%    - - - - -
% 3 |0|0|0|0|0|
%    - - - - -
% 2 |0|0|0|0|0|
%    - - - - -
% 1 |0|0|0|0|0|
%    - - - - -
%    1 2 3 4 5

persistent centers

% Compute centers only once
if isempty(centers)
    dim_state = size(range,1);
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
    distance = bsxfun(@minus,state',reshape(centers,[1 size(centers)])).^2;
    distance = sqrt(sum(distance,2));
    Phi = zeros(size(centers,2),size(state,2));
    [~, idx] = min(distance,[],3);
    idx = sub2ind(size(Phi), idx', 1:size(Phi,2));
    Phi(:) = 0;
    Phi(idx) = 1;

    if offset == 1, Phi = [ones(1,size(state,2)); Phi]; end
end
