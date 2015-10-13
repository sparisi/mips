function phi = deep_basis_tile2(state)

numfeatures = 11;

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numfeatures;
    return
end

% This is a more generic tile coding, with each tile being a 4x4 cell. Each
% tile has the border cells overlapping with its neighbors. In addition,
% there is a 1x1 tile corresponding to the first cell.
% The last row (10) is not covered, as there is only one blank cell and the
% optimal action is to always go down.

[d,n] = size(state);
assert(d == 2);
phi = zeros(numfeatures,n);

phi(1,:) = 1;
phi(2,:) = ( (state(1,:) >= 1 & state(1,:) <= 4) & ...
    (state(2,:) >= 1 & state(2,:) <= 4) & ~(state(1,:) == 1 & state(2,:) == 1) );
phi(3,:) = ( (state(1,:) >= 4 & state(1,:) <= 7) & ...
    (state(2,:) >= 1 & state(2,:) <= 4) );
phi(4,:) = ( (state(1,:) >= 7 & state(1,:) <= 10) & ...
    (state(2,:) >= 1 & state(2,:) <= 4) );
phi(5,:) = ( (state(1,:) >= 1 & state(1,:) <= 4) & ...
    (state(2,:) >= 4 & state(2,:) <= 7) );
phi(6,:) = ( (state(1,:) >= 4 & state(1,:) <= 7) & ...
    (state(2,:) >= 4 & state(2,:) <= 7) );
phi(7,:) = ( (state(1,:) >= 7 & state(1,:) <= 10) & ...
    (state(2,:) >= 4 & state(2,:) <= 7) );
phi(8,:) = ( (state(1,:) >= 1 & state(1,:) <= 4) & ...
    (state(2,:) >= 7 & state(2,:) <= 10) );
phi(9,:) = ( (state(1,:) >= 4 & state(1,:) <= 7) & ...
    (state(2,:) >= 7 & state(2,:) <= 10) );
phi(10,:) = ( (state(1,:) >= 7 & state(1,:) <= 10) & ...
    (state(2,:) >= 7 & state(2,:) <= 10) );
phi(11,:) = ( state(1,:) == 1 & state(2,:) == 1 );

return
