function phi = dam_basis_rbf(state)

% n_centers = 4;
% range = [0,160];
% if nargin == 0
%     phi = basis_krbf(n_centers,range,0);
% else
%     assert(size(state,1) == 1);
%     phi = basis_krbf(n_centers,range,0,state);
% end
    
n_centers = 4;
range = [-20 190];
width = 60;
if nargin == 0
    phi = basis_rrbf(n_centers,width,range,0);
else
    assert(size(state,1) == 1);
    phi = basis_rrbf(n_centers,width,range,0,state);
end
