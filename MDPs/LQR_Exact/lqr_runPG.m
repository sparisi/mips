%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single-objective Natural Policy Gradient optimization for a 
% multidimensional LQR.
% The policy is Gaussian with linear mean and constant covariance.
% Only the mean is learned.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 2;
LQR = lqr_init(N);

g = LQR.g;
Q = LQR.Q{1};
R = LQR.R{1};
A = LQR.A;
B = LQR.B;
x0 = LQR.x0;
Sigma = eye(N);
K = -0.5 * eye(N);

lrate = 0.1;
learnIter = 1000;
for i = 1 : learnIter
    [W1, W2] = lqr_natural(A,B,Q,R,K,Sigma,g);
    K = K - lrate * W1;
    Sigma = Sigma - lrate * W2;
end

J = lqr_return(A,B,Q,R,K,Sigma,x0,g);

[X,L,G] = dare(A,B,Q,R);
error = abs(G) - abs(K)
