function areEq = areAlmostEqual( A, B, tolerance )
% Checks if two matrices are almost equal.
%
% Inputs: 
% - A         : matrix (any size)
% - B         : matrix (same size of A)
% - tolerance : a threshold of tolerance
%
% Outputs:
% - areEq     : 1 if c(i,j,..) < tolerance for all (i,j,...), 
%               where C = |A - B|, 0 otherwise

areEq = max(abs(A(:)-B(:))) < tolerance;

end
