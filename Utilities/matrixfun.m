function out = matrixfun(fun,M,dim)
% MATRIXFUN Applies function FUN to every row (if DIM == 1) or column (if 
% DIM == 2) of M.

assert(dim == 1 || dim == 2);

% because of how cell2fun interprets the input
dims = [2 1];
dim = dims(dim);
C = num2cell(M,dim);
out = cellfun(fun,C,'UniformOutput',false);

end