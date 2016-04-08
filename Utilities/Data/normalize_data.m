function pn = normalize_data(p,minp,maxp)
% NORMALIZE_DATA Normalizes data points by pn = (p - minp) / (maxp - minp).
% If MINP == min(P) and MAXP == max(P), the points are normalized in [0,1].
%
%    INPUT
%     - p    : [N x D] matrix, where N is the number of points and D is the
%              dimensionality of a point
%     - minp : (optional) [1 x D] vector of the minimum feasible value the 
%              points can assume (min(p) by default)
%     - maxp : (optional) [1 x D] vector of the maximum feasible value the 
%              points canassume (max(p) by default)
%
%    OUTPUT
%     - pn   : [N x D] matrix of normalized points

if nargin == 1
    minp = min(p);
    maxp = max(p);
end

% checkmin = bsxfun(@ge,p,minp);
% checkmin = min(checkmin(:));
% checkmax = bsxfun(@le,p,maxp);
% checkmax = min(checkmax(:));
% if ~checkmin || ~checkmax
%     warning('There are points out of the normalizing bounds.')
% end

pn = bsxfun(@times,bsxfun(@plus,p,-minp),1./(maxp-minp));

end