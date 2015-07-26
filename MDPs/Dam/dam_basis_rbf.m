function phi = dam_basis_rbf(state)

% n_centers = 4;
% range = [0,160];
% if nargin == 0
%     phi = basis_krbf(n_centers,range) + 1;
% else
%     phi = [1; basis_krbf(n_centers,range,state)];
% end
    
n_centers = 4;
range = [-20,190];
width = 60;
if nargin == 0
    phi = basis_rrbf(n_centers,width,range) + 1;
else
    phi = [1; basis_rrbf(n_centers,width,range,state)];
end
    
% nr_centers = 4;
% centers = [0; 50; 120; 160];
% widths = [50; 20; 40; 50];
% 
% if nargin == 0
%     phi = nr_centers + 1;
% else
%     phi = zeros((size(centers, 1) + 1), 1);
%     
%     index = 1;
%     phi(index) = 1;
%     for i = 1 : size(centers, 1)
%         phi(index + i) = exp(-norm(state - centers(i)) / widths(i));
%     end
% end

return;
