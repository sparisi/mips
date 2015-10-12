function rw = weightedScalarization(r,w,metric,u)
% SCALARIZEREWARD Scalarizes rewards according to a weighted function.
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

switch metric
    case 'ws'
        rw = sum(bsxfun(@times, w/sum(w), r), 2);
    case 'chebychev'
        rw = bsxfun(@(x,y)((x-y).^2), r, u);
        rw = bsxfun(@times, rw, w/sum(w));
        rw = -max(rw,[],2);
    case 'localutopia'
        rw = bsxfun(@(x,y)((x-y).^2), r, w.*u);
        rw = -sqrt(sum(rw,2));
    otherwise
        error('Unknown scalarization function.')
end

end