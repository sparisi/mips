function [H, i, mse] = apg(X1, X2, Y, H, varargin)
% APG Accelerated Proximal Gradient. Learns a model 
% Y = [ X1, X2, 1 ] H [ X1, X2, 1 ]'
% where
%     [ R1      R2      0.5r1 ]
% H = [ R2'     Rc      0.5r2 ]
%     [ 0.5r1'  0.5r2'  r0    ]
% It also performs regularization on R2. The regularization can be based on
% l1-norm, l2-norm or nuclear norm.

[d1, N] = size(X1);
[d2, N] = size(X2);
X = [X1; X2; ones(1,N)];

options = {'lambda_l2', 'lambda_nn', 'lambda_l1', 'lrate', 'maxiter'};
defaults = {0, 0, 0, 0.001, 300};
[lambda_l2, lambda_nn, lambda_l1, lrate, maxiter] = ...
    internal.stats.parseArgs(options, defaults, varargin{:});

assert(~(lambda_nn > 0 && lambda_l1 > 0), ...
    'Cannot perform both nuclear norm and l1-norm regularization.')

t = 1; % Extrapolation parameters
t_prev = t;
H_prev = H;

if nargout == 3, mse(1) = mean((sum((X'*H)'.*X)-Y).^2); end

for i = 1 : maxiter
    
    % Adaptive stepsize
%     lrate = lrate/i;
    
    H_ex = H + (t_prev-1)/t * (H-H_prev); % Extrapolated solution
    
    % Gradient update
    G = bsxfun(@times, sum((X'*H_ex)'.*X,1)-Y, X) * X';
    G = G/N + lambda_l2 * H_ex; % L2 regularization
    G = (G+G')/2; % Enforce symmetry
    G_fnorm = norm(G,'fro');
    
    H_ex = H_ex - lrate/G_fnorm*G;
    
    R1 = H_ex(1:d1,1:d1);
    R2 = H_ex(d1+1:d1+d2, d1+1:d1+d2);
    Rc = H_ex(1:d1, d1+1:d1+d2);
    
    % Nuclear norm regularization on R2 and Rc
    if lambda_nn ~= 0
        [U,S,V] = svd(R2);
        H_ex(d1+1:d1+d2, d1+1:d1+d2) = U * max(S-lambda_nn*eye(size(S)),0) * V';
%         [U,S,V] = svd(Rc);
%         Rc = U * max(S-lambda_nn*eye(size(S)),0) * V';
%         H_ex(1:d1, d1+1:d1+d2) = Rc;
%         H_ex(d1+1:d1+d2, 1:d1) = Rc';
    end
    
    % L1 regularization on R2 and Rc
    if lambda_l1 ~= 0
        H_ex(d1+1:d1+d2, d1+1:d1+d2) = max(abs(R2) - lambda_l1, 0) .* sign(R2);
%         H_ex(1:d1, d1+1:d1+d2) = max(abs(Rc) - lambda_l1, 0) .* sign(Rc);
%         H_ex(d1+1:d1+d2, 1:d1) = H_ex(1:d1, d1+1:d1+d2)';
    end

    % Enforce R1 to be negative definite
%     H_ex(1:d1,1:d1) = -nearestSPD(-R1);
    [U,V] = eig(R1);
    V(V>0) = -1e-8;
    H_ex(1:d1,1:d1) = U * V * U';

    % Terminal conditions
%     if G_fnorm < 1e-8, break, end
%     if norm(H_ex-H_prev,'fro') < 1e-6, break, end
    if nnz(H_ex(d1+1:d1+d2, d1+1:d1+d2)) == 0, break, end
    
    % Update
    H_prev = H;
    t_prev = t;
    H = H_ex;
    t = (1 + sqrt(1+4*t^2))/2;
    
end

H = H_ex;

if nargout == 3, mse(2) = mean((sum((X'*H)'.*X)-Y).^2); end
