function invM = inv3(M)
% INV3 Inverts every matrix M(:,:,i) in a 3d matrix.

[rows, cols, pages] = size(M);
assert(rows == cols, 'Matrices must be square.')

% Reshape M as a square sparse matrix
I = reshape(1:rows*pages,rows,1,pages);
I = repmat(I,[1 cols 1]);
J = reshape(1:cols*pages,1,cols,pages);
J = repmat(J,[rows 1 1]);
M = sparse(I(:),J(:),M(:));

% Invert M
invM = M \ repmat(eye(rows),[pages,1]);

% Reshape to the original dimensions
invM = reshape(invM, [cols pages rows]);
invM = permute(invM,[1 3 2]);
invM = reshape(invM,[cols rows pages]);

end