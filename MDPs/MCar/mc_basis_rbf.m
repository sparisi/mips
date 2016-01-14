function phi = mc_basis_rbf(state)

n_centers = 4;
range = [-1 1; -3 3];

if nargin == 0
    phi = basis_krbf(n_centers,range);
else
    assert(size(state,1) == 2);
    phi = basis_krbf(n_centers,range,state);
end

end