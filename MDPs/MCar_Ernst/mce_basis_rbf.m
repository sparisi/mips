function phi = mce_basis_rbf(state)

n_centers = 4;
range = [-2 1; -4 4];

if nargin == 0
    phi = basis_krbf(n_centers,range) + 1;
else
    assert(size(state,1) == 2);
    phi = [ones(1,size(state,2)); basis_krbf(n_centers,range,state)];
end

end