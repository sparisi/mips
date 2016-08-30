function rw = metric_cheby(r,w,u)
% METRIC_CHEBY Scalarizes a set of rewards according to their Chebychev
% distance from a utopia point.
%
%    INPUT
%     - r  : [N x D] rewards matrix, where N is the number of reward
%            samples and D is the dimension of the reward vector
%     - w  : [1 x D] vector with the weights for the scalarization
%     - u  : [1 x D] utopia point
%
%    OUTPUT
%     - rw : [N x 1] scalarized rewards matrix

rw = abs(bsxfun(@minus, r, u));
rw = bsxfun(@times, rw, w/sum(w));
rw = -max(rw,[],2);
