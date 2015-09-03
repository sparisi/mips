function areEq = areAlmostEqual(A, B, tolerance)
% AREALMOSTEQUAL Checks if two matrices are almost equal.
%
%    INPUT
%     - A         : matrix (any size)
%     - B         : matrix (same size of A)
%     - tolerance : a threshold of tolerance
%
%    OUTPUT
%     - areEq     : 1 if c(i,j,..) < tolerance for all (i,j,...), 
%                   where C = |A - B|, 0 otherwise

areEq = max(abs(A(:)-B(:))) < tolerance;

end
