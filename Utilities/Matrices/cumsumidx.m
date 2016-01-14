function out = cumsumidx(v, i)
% CUMSUMIDX Perform an indices-wise cumulative sum over a matrix. The sum
% is performed along the second dimension.
% 
% =========================================================================
% EXAMPLE
% v = [1:10; 11:20]; i = [3 6 10];
% In this case we want the cumulative sum of v(:,1:3), v(:,4:6), v(:,7:10).
% Therefore, the result is [6 15 34; 36 45 74].

assert(length(i) <= size(v,2), ...
    'The number of indices is higher than the number of elements.')

c = cumsum(v,2);
r = c(:,i);
out = [c(:,i(1)) diff(r,[],2)];