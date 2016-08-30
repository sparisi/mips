function rw = metric_ws(r,w)
% METRIC_WS Scalarizes a set of rewards according to a linear weighted sum.
%
%    INPUT
%     - r      : [N x D] rewards matrix, where N is the number of reward 
%                samples and D is the dimension of the reward vector
%     - w      : [1 x D] vector with the weights for the scalarization
%
%    OUTPUT
%     - rw     : [N x 1] scalarized rewards matrix

rw = sum(bsxfun(@times, w/sum(w), r), 2);
