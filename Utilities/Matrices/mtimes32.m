function C = mtimes32(A,B)
% MTIMES32 Multiplies a 3d matrix by a 2d matrix.
% It is equivalent to the following loop:
% >> C = zeros(size(B));
% >> for i = 1 : size(B,2)
% >>     C(:,i) = A(:,:,i) * B(:,i);
% >> end
%
%    INPUT
%     - A : [D x D x N] matrix
%     - B : [D x N] matrix
%
%    OUTPUT
%     - C : [D x N] matrix

C = bsxfun(@times,A,reshape(B,[1 size(B)]));
C = permute(sum(C,2),[1 3 2]);
