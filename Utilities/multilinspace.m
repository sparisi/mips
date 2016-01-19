function y = multilinspace(n, varargin)
% MULTILINSPACE Like linspace, but with several intervals.
%
% =========================================================================
% EXAMPLE
% d1 = [1 5]; d2 = [1 10]; d3 = [0 -10];
% y = multilinspace(5,d1,d2,d3)
%    1.0000    2.0000    3.0000    4.0000    5.0000
%    1.0000    3.2500    5.5000    7.7500   10.0000
%         0   -2.5000   -5.0000   -7.5000  -10.0000

y = cellfun(@(x)linspace(x(1),x(2),n),varargin,'uni',0);
y = vertcat(y{:});
