function phi = damctx_basis_rbf(state)
    
n_centers = 6;
range = [-20 320];
width = 60;
if nargin == 0
    phi = basis_rrbf(n_centers,width,range);
else
    assert(size(state,1) == 1);
    phi = basis_rrbf(n_centers,width,range,state);
end
