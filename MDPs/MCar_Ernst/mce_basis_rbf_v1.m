function phi = mce_basis_rbf_v1(state,action)

n_centers = 4;
if nargin == 0
    phi = 3 * (basis_krbf(n_centers, [-2 1; -4 4]) + 1);
else
    Phi = [1; basis_krbf(n_centers, [-2 1; -4 4], state)];
    dim_phi = length(Phi);
    phi = zeros(3*dim_phi,1);
    i = action - 1;
    phi(dim_phi*i+1 : dim_phi*i+dim_phi) = Phi;
end 

end