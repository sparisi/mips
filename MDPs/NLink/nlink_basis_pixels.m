function phi = nlink_basis_pixels(mdp, varargin)

if nargin < 2
    phi = mdp.render();
else
    phi = mdp.render(varargin{:});
    phi = reshape(phi,[size(phi,1)*size(phi,2),size(phi,3)]);
end
