function y = multigrid(n, varargin)
% MULTIGRID Produces a multi-dimensional grid. Each dimension is split by 
% LINSPACE and then all linear spaces are meshed.
% 
%    INPUT
%     - N           : number of points per dimension (argument of LINSPACE)
%     - X1, X2, ... : 2-dimensional vectors denoting the bounds for each
%                     dimension
%
%    OUTPUT
%     - Y           : [M x N^M] matrix, where M is the number of input X
%
% =========================================================================
% EXAMPLE
% d1 = [1 3]; d2 = [-5 -1]; d3 = [10 20];
% y = multigrid(3,d1,d2,d3)
%   1  2  3  1  2  3  1  2  3  1  2  3  1  2  3  1  2  3  1  2  3  1  2  3  1  2  3
%  -5 -5 -5 -3 -3 -3 -1 -1 -1 -5 -5 -5 -3 -3 -3 -1 -1 -1 -5 -5 -5 -3 -3 -3 -1 -1 -1

y = cellfun(@(x)linspace(x(1),x(2),n),varargin,'uni',0);
c = cell(1,length(varargin{1}));
[c{:}] = ndgrid(y{:});
y = cell2mat( cellfun(@(v)v(:), c, 'UniformOutput', false) )';
