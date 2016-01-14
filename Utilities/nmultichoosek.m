function combs = nmultichoosek(values, k)
% NMULTICHOOSEK Like nchoosek, but with repetitions. The VALUES for which
% nchoosek is performed are columns. If VALUES is a matrix, nchoosek is
% performed for each column and COMBS is a matrix as well.

[n, ncombs] = size(values);
if n == 1
    n = values;
    combs = nchoosek(n+k-1, k);
else
    combs = bsxfun(@minus, nchoosek(1:n+k-1,k), 0:k-1);
    combs = reshape(values(combs,:), [], k, ncombs);
end