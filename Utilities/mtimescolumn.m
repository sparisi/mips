function C = mtimescolumn(A, B)
% MTIMESCOLUMN Multiplies each column of a matrix A by each row of a matrix
% B to obtain many 2d matrices. These matrices are then vectorized in a 2d
% matrix (one vectorization per column).
% It is equivalent to the following loop:
% >> for i = 1 : D
% >>     tmp = A(:,i) * B(:,i);
% >>     C(:,i) = tmp(:);
% >> end
%
%    INPUT
%     - A : N-by-D matrix
%     - B : M-by-D matrix
%
%    OUTPUT
%     - C : N*M-by-D matrix

[N, D] = size(A);
M = size(B,1);

A = repmat(A, M, 1);
B = kron(B, ones(N,1));
C = reshape(A(:) .* B(:), N*M, D);