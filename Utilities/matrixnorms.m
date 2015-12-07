function norms = matrixnorms(M,dim)
% MATRIXNORMS Computes the norm of each row (DIM = 1) or column (DIM = 2)
% of a matrix M.

norms = sqrt(sum(M.^2,dim));
