function y = quadcostmulti(x, xdes, C)
% Multimodal quadratic cost: y = max_i [ -(x-xdes_i)'*C*(x-xdes_i) ].

if nargin < 3
    d = size(x,1);
    C = eye(d);
end

for i = 1 : size(xdes,2)
    diff = bsxfun(@minus,x,xdes(:,i));
    y(i,:) = -sum(bsxfun(@times, diff'*C, diff'), 2)';
end

y = max(y,[],1);