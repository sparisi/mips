function phi = puddle_basis_tile(state)

numfeatures = 7;

% If no arguments just return the number of basis functions
if nargin == 0
    phi = numfeatures;
    return
end

phi = zeros(numfeatures,1);
phi(1) = 1;
phi(2) = (state(1)+0.1);
phi(3) = (state(2)+0.1);
phi(4) = (state(1) > 0.95);
phi(5) = (state(2) > 0.95);
phi(6) = (state(2) >= 0.75 && state(1) < 0.45);
phi(7) = (state(1) >= 0.45);

return
