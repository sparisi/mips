function phi = mce_basis_rbf(state)

env = mce_environment();
n_centers = 4;
range = [env.xLB env.xUB; env.vLB env.vUB];

if nargin == 0
    phi = basis_krbf(n_centers,range) + 1;
else
    assert(size(state,1) == 2);
    phi = [ones(1,size(state,2)); basis_krbf(n_centers,range,state)];
end

end