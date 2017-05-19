function d = L2_distance_sq(a, b)
% L2_DISTANCE_SQ Squared L2 distance between A (D x M matrix) and B (D x N
% matrix).

d = bsxfun(@plus, sum(a .* a)', bsxfun(@minus, sum(b .* b), 2 * a' * b));