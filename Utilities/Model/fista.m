function [H, i, mse] = fista(X1, X2, Y, H, varargin)
% FISTA Fast Iterative Shrinkage-Thresholding Algorithm. 
% Learns a model Y = [ X1, X2, 1 ] H [ X1, X2, 1 ]'
% where
%     [ R1      Rc      0.5r1 ]
% H = [ Rc'     R2      0.5r2 ]
%     [ 0.5r1'  0.5r2'  r0    ]
% It also performs regularization on R2. The regularization can be based on
% l1-norm or nuclear norm.

[d1, N] = size(X1);
[d2, N] = size(X2);
X = [X1; X2; ones(1,N)];

p = inputParser;
p.KeepUnmatched = true;
addOptional(p, 'lambda_nn', 0);
addOptional(p, 'lambda_l1', 0);
addOptional(p, 'lambda_l2', 0);
addOptional(p, 'maxiter', 300);
addOptional(p, 'lrate', 0.00001);
parse(p,varargin{:});

lambda_l2 = p.Results.lambda_l2;
lambda_l1 = p.Results.lambda_l1;
lambda_nn = p.Results.lambda_nn;
maxiter = p.Results.maxiter;
lrate = p.Results.lrate;

assert(~(lambda_nn > 0 && lambda_l1 > 0), ...
    'Cannot perform both nuclear norm and l1-norm regularization.')

t = 1; % Extrapolation parameter

if nargout == 3, mse(1) = mean((sum((X'*H)'.*X)-Y).^2); end

for i = 1 : maxiter

    H_prev = H;
    t_prev = t;
    if i == 1, t_prev = 0; end
    
    % Gradient update
    G_error = (bsxfun(@times, sum((X'*H)'.*X,1)-Y, X) * X') / N; % Error derivative
    G_fro = lambda_l2 * H; % Frobenius norm derivative
    G = G_error + G_fro;
    G = (G+G')/2; % Enforce symmetry
    fnorm = norm(G,'fro'); % Frobenius norm to normalize the gradient
    H = H - lrate/fnorm*G;
    
    % Nuclear norm proximal operator on R2
    if lambda_nn > 0
        H(d1+1:d1+d2, d1+1:d1+d2) = proximal_nn(H(d1+1:d1+d2, d1+1:d1+d2), lambda_nn);
    end
    
    % L1-norm proximal operator on R2
    if lambda_l1 > 0
        H(d1+1:d1+d2, d1+1:d1+d2) = proximal_l1(H(d1+1:d1+d2, d1+1:d1+d2), lambda_l1);
    end

    % Enforce R1 to be negative definite
    R1 = H(1:d1,1:d1);
    [U,V] = eig(R1);
    V(V>0) = -1e-8;
    H(1:d1,1:d1) = U * V * U';

    % Terminal condition
    if 1/(lrate/fnorm) * norm((H - H_prev)) < 0.001, break, end
    
    % Update
    t = (1 + sqrt(1+4*t^2))/2;
    H = H + (t_prev-1)/t * (H-H_prev);
    
end

if nargout == 3, mse(2) = mean((sum((X'*H)'.*X)-Y).^2); end
