function rw = scalarizeReward(r,w,metric,u)
% SCALARIZEREWARD Scalarizes rewards according to a function.
%
%    INPUT
%     - r      : N-by-D matrix, where N is the number of reward samples and D
%                is the dimension of the reward vector
%     - w      : D-by-1 vector with the weights for the scalarization
%     - metric : string defining the scalarization function
%     - u      : utopia point (needed by only some metrics)
%
%    OUTPUT
%     - rw     : normalized reward matrix

if strcmp('ws',metric)
    rw = sum(bsxfun(@times, w/sum(w), r), 2);
elseif strcmp('chebychev',metric)
    rw = bsxfun(@(x,y)((x-y).^2), r, u);
    rw = bsxfun(@times, rw, w/sum(w));
    rw = -max(rw,[],2);
elseif strcmp('localutopia',metric)
    rw = bsxfun(@(x,y)((x-y).^2), r, w.*u);
    rw = -sqrt(sum(rw,2));
else
    error('Unknown scalarization function.')
end

end