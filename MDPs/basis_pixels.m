function phi = basis_pixels(mdp, varargin)
% The MDP requires a function 'render' that returns the pixels.

if nargin < 2
    phi = mdp.render();
else
    phi = mdp.render(varargin{:});
    phi = reshape(phi,[size(phi,1)*size(phi,2),size(phi,3)]);
end
