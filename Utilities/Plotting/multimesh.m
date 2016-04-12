function multimesh(f,xmin,xmax,ymin,ymax,n)
% MULTIMESH Mesh for functions with vectorial input and output.
%
%    INPUT
%     - f    : function handle
%     - xmin : lower bound on x
%     - xmax : upper bound on x
%     - ymin : lower bound on y
%     - ymax : upper bound on y
%     - n    : (optional) number of points for meshing
%
% =========================================================================
% EXAMPLE
% f = @(varargin)basis_krbf(7, [0 1; 0 1], varargin{:});
% multimesh(f,0,1,0,1)

if nargin < 6, n = 50; end

x = linspace(xmin,xmax,n);
y = linspace(ymin,ymax,n);
[X, Y] = meshgrid(x,y);
Z = f([X(:), Y(:)]');
Z = reshape(Z',n,n,size(Z,1));

figure, hold all
for i = 1 : size(Z,3)
    mesh(X,Y,Z(:,:,i))
end

axis([xmin,xmax,ymin,ymax])

end