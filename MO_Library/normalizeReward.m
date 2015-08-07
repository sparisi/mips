function rn = normalizeReward(r,minr,maxr)
% Normalizes rewards in order to have them in [0,1]. The normalization is
% rn = (r - rmax) / (rmin - rmax)
%
% Inputs:
% - r    : N-by-D matrix, where N is the number of reward vectors samples 
%          and D is the number of rewards
% - minr : 1-by-D vector of the minimum feasible value the rewards can have
% - maxr : 1-by-D vector of the maximum feasible value the rewards can have

assert(isvector(minr));
assert(isvector(maxr));
assert(size(r,2) == length(minr));
assert(size(r,2) == length(maxr));

checkmin = bsxfun(@ge,r,minr);
checkmin = min(checkmin(:));
checkmax = bsxfun(@le,r,maxr);
checkmax = min(checkmax(:));

if ~checkmin || ~checkmax
    warning('Rewards greater or less than maximum and minimum boundaries.')
end

rn = bsxfun(@times,bsxfun(@plus,r,-minr),1./(maxr-minr));

end