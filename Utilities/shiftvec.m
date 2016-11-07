function y = shiftvec(x, idx, idxmax)
% SHIFTVEC Shifts vectors X according to different starting rows.
%
% =========================================================================
% EXAMPLE
%     [ 1 4 ]
% X = [ 2 5 ] , idx = [1 2], idxmax = 3
%     [ 3 6 ]
%
% shiftvec(X,idx,idxmax)
%
%     1     0
%     2     0
%     3     0
%     0     4
%     0     5  <-- [1 2 3] and [4 5 6] have been shifted according to IDX
%     0     6
%     0     0
%     0     0  <-- these zeros are due to IDXMAX
%     0     0

[dim, n] = size(x);
idx_start = (idx-1)*dim+1 + (0:n-1)*dim*idxmax; % Starting linear indices
idx = bsxfun(@plus,idx_start,(0:dim-1)'); % All linear indices
y = zeros(dim*idxmax,n); % Initialize output array with zeros
y(idx) = x;
