function rw = metric_localutopia(r,w,u,au)
% METRIC_LOCALUTOPIA Scalarizes a set of rewards according to a local 
% utopia point.
%
%    INPUT
%     - r  : [N x D] rewards matrix, where N is the number of reward
%            samples and D is the dimension of the reward vector
%     - w  : [1 x D] vector with the weights for the scalarization,
%            representing the local-utopia
%     - u  : [1 x D] vector representing the utopia point
%     - au : [1 x D] vector representing the antiutopia point
%
%    OUTPUT
%     - rw : [N x 1] scalarized rewards matrix

r = normalize_data(r,au,u);
rw = bsxfun(@minus,r,w);
rw = -matrixnorms(rw,1).^2;
