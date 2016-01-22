function rw = metric_ws(r,w)
% METRIC_WS Scalarizes a set of rewards according to a linear weighted sum.
%
%    INPUT
%     - r      : [D x N] rewards matrix, where N is the number of reward 
%                samples and D is the dimension of the reward vector
%     - w      : [D x 1] vector with the weights for the scalarization
%
%    OUTPUT
%     - rw     : scalarized rewards matrix

rw = sum(bsxfun(@times, w/sum(w), r), 1);
