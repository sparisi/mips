function flag = issingular(A)
% ISSINGULAR Checks singularity of a matrix.

[n,m] = size(A);
flag = rank(A) < min(n,m);