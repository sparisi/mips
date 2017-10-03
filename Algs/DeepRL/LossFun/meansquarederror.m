function [l, dl] = meansquarederror(y, t)
% MEANSQUAREDERROR Mean squared error between predictions Y and targets T (both D x N 
% matrices, where D is the dimensionality and N is the number of samples).

e = t - y;
l = 0.5 * mean(e.^2, 2);
dl = -e;
