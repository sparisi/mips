function rw = metric_cheby(r,w,u)
% METRIC_CHEBY Scalarizes a set of rewards according to their Chebychev
% distance from an utopia point.
%
%    INPUT
%     - r  : [D x N] rewards matrix, where N is the number of reward
%            samples and D is the dimension of the reward vector
%     - w  : [D x 1] vector with the weights for the scalarization
%     - u  : [D x 1] utopia point
%
%    OUTPUT
%     - rw : scalarized rewards matrix

rw = abs(bsxfun(@minus, r, u));
rw = bsxfun(@times, rw, w / sum(w));
rw = -max(rw,[],1);
