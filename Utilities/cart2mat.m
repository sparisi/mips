function [x,y] = cart2mat(i,j,m)
% CART2MAT Given a matrix of size [M x N], this functions converts the 
% matrix coordinates (i,j) to cartesian coordinates (x,y).
%
% (y)
%  5
%  4
%  3     CART
%  2
%  1
%    1 2 3 4 5 (x)
%
%        TO
%
%    1 2 3 4 5 (j)
%  1
%  2
%  3     MAT
%  4
%  5 
% (i)

idx = -(-m:-1);
y = idx(i);
x = j;
