function pn = normalize_points(p,minp,maxp)
% NORMALIZE_POINTS Normalizes points in order to have them in [0,1].
% pn = (p - minp) / (maxp - minp)
%
%    INPUT
%     - p    : [N x D] matrix, where N is the number of points and D is the
%              dimensionality of a point
%     - minp : [1 x D] vector of the minimum feasible value the points can
%              assume
%     - maxp : [1 x D] vector of the maximum feasible value the points can
%              assume
%
%    OUTPUT
%     - pn   : [N x D] matrix of normalized points

% checkmin = bsxfun(@ge,p,minp);
% checkmin = min(checkmin(:));
% checkmax = bsxfun(@le,p,maxp);
% checkmax = min(checkmax(:));
% if ~checkmin || ~checkmax
%     warning('There are points out of the normalizing bounds.')
% end

pn = bsxfun(@times,bsxfun(@plus,p,-minp),1./(maxp-minp));

end