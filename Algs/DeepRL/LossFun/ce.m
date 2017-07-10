function [l, dl] = ce(y, t)
% CE Cross entropy between predictions Y and targets T (both D x N 
% matrices, where D is the dimensionality and N is the number of samples).

l = -mean(t.*log(y) + (1-t).*log(1-y),2);
dl = (y-t) ./ ( (1-y).*y );