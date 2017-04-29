function [H, i, mse] = apg(X1, X2, Y, H, varargin)
% APG Accelerated Proximal Gradient Method with Non-Monotonic Backtracking. 
% Learns a model Y = [ X1, X2, 1 ] H [ X1, X2, 1 ]'
% where
%     [ R1      Rc      0.5r1 ]
% H = [ Rc'     R2      0.5r2 ]
%     [ 0.5r1'  0.5r2'  r0    ]
% It also performs regularization on R2. The regularization can be based on
% l1-norm or nuclear norm.
%
% =========================================================================
% REFERENCE
% N Ito, A Takeda, K Toh
% A Unified Formulation and Fast Accelerated Proximal Gradient Method for 
% Classification (2017)

[d1, N] = size(X1);
[d2, N] = size(X2);
X = [X1; X2; ones(1,N)];

p = inputParser;
p.KeepUnmatched = true;
addOptional(p, 'lambda_nn', 0);
addOptional(p, 'lambda_l1', 0);
addOptional(p, 'lambda_l2', 0);
addOptional(p, 'maxiter', 300);
parse(p,varargin{:});

lambda_l2 = p.Results.lambda_l2;
lambda_l1 = p.Results.lambda_l1;
lambda_nn = p.Results.lambda_nn;
maxiter = p.Results.maxiter;

assert(~(lambda_nn > 0 && lambda_l1 > 0), ...
    'Cannot perform both nuclear norm and l1-norm regularization.')

if nargout == 3, mse(1) = mean((sum((X'*H)'.*X)-Y).^2); end

t = 1;
t_prev = 0;
alpha = H;
alpha_prev = alpha;
beta = H;
eta_u = 10;
eta_d = 10;
L = 1;
L_prev = L;
epsilon = 0.001;

F = @(H) func(X,Y,H,lambda_l2,lambda_l1,lambda_nn,d1,d2);
Q = @(H,H_ex,L) approx(X,Y,H,H_ex,lambda_l2,lambda_l1,lambda_nn,L,d1,d2);
T = @(H,L) prox(X,Y,H,lambda_l2,lambda_l1,lambda_nn,L,d1,d2);

for i = 1 : maxiter
    
    alpha_prev_prev = alpha_prev;
    alpha_prev = alpha;
    
    [alpha, fnorm] = T(beta,L);
    
    while true
        if F(alpha) <= Q(alpha,beta,L), break, end
        L = eta_u * L;
        t = (1 + sqrt(1+4*(L/L_prev)*t_prev^2))/2;
        beta = alpha_prev + (t_prev-1)/t * (alpha_prev - alpha_prev_prev);
        [alpha, fnorm] = T(beta,L);
    end
    
    if L * fnorm * norm(T(alpha,L) - alpha) < epsilon, break, end
    
    L_prev = L;
    L = L / eta_d;
    t_prev = t;
    t = (1 + sqrt(1+4*(L/L_prev)*t^2)) / 2;
    beta = alpha + (t_prev-1)/t * (alpha - alpha_prev);
end

H = alpha;

% Enforce R1 to be negative definite
R1 = H(1:d1,1:d1);
[U,V] = eig(R1);
V(V>0) = -1e-8;
H(1:d1,1:d1) = U * V * U';

if nargout == 3, mse(2) = mean((sum((X'*H)'.*X)-Y).^2); end


%%
function F = func(X,Y,H,lambda_l2,lambda_l1,lambda_nn,d1,d2)

R2 = H(d1+1:d1+d2, d1+1:d1+d2);
F = mean((sum((X'*H)'.*X)-Y).^2) + lambda_l2 * norm(H,'fro')^2 + ...
        lambda_nn * trace(sqrt(R2'*R2)) + lambda_l1 * norm(R2,1);
    
%%
function Q = approx(X,Y,H,H_ex,lambda_l2,lambda_l1,lambda_nn,L,d1,d2)

R2 = H(d1+1:d1+d2, d1+1:d1+d2);
dF_ex = (bsxfun(@times, sum((X'*H_ex)'.*X,1)-Y, X) * X') / length(Y) + lambda_l2 * H_ex;
Q = mean((sum((X'*H_ex)'.*X)-Y).^2) + lambda_l2 * norm(H_ex,'fro')^2 + ...
    dot(dF_ex(:), H(:) - H_ex(:)) + L/2 * norm(H(:)-H_ex(:))^2 + ...
    lambda_nn * trace(sqrt(R2'*R2)) + lambda_l1 * norm(R2,1);

%%
function [H, fnorm] = prox(X,Y,H,lambda_l2,lambda_l1,lambda_nn,L,d1,d2)

G = (bsxfun(@times, sum((X'*H)'.*X,1)-Y, X) * X') / length(Y) + lambda_l2 * H;
G = (G+G')/2;
fnorm = norm(G,'fro');
H = H - 1/L/fnorm*G;

if lambda_nn > 0
    H(d1+1:d1+d2, d1+1:d1+d2) = proximal_nn(H(d1+1:d1+d2, d1+1:d1+d2), lambda_nn);
end

if lambda_l1 > 0
    H(d1+1:d1+d2, d1+1:d1+d2) = proximal_l1(H(d1+1:d1+d2, d1+1:d1+d2), lambda_l1);
end