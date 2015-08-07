function out = matrixfun(M,fun,dim)
% Applies function FUN to every row (if DIM == 1) or column (if DIM == 2)
% of M.

assert(dim == 1 || dim == 2);
C = num2cell(M,dim);
out = cellfun(fun,C,'UniformOutput',false);

end