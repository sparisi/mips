function vec = ind2vec(ind, n)
% IND2VEC Converts a vector of indices to a matrix of 0 / 1.
%
%    INPUT
%     - ind : vector of indices
%     - n   : number of columns of the output matrix
%
%    OUTPUT
%     - vec : matrix of vectors representing the indices
%
% =========================================================================
% EXAMPLE
% ind = [1 1 2 1 1 2 2 3 1]'; n = 3;
% vec = ind2vec(ind, n)
%
%     1     0     0
%     1     0     0
%     0     1     0
%     1     0     0
%     1     0     0
%     0     1     0
%     0     1     0
%     0     0     1
%     1     0     0

assert(isrow(ind) || iscolumn(ind), 'Input must be a vector')

if isrow(ind), ind = ind'; end

s = [size(ind,1),n];
vec = zeros(s);
vec(sub2ind(s,1:s(1),ind')) = 1;
