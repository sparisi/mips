function [U, S, V] = rsvds(A, k)
% RSVDS Randomized SVDS.
%
% =========================================================================
% REFERENCE
% H Halko, P G Martinsson, J A Tropp
% Finding Structure with Randomness: Probabilistic Algorithms for 
% Constructing Approximate Matrix Decompositions (2011)

[m,n]    = size(A);
p        = min(2*k,n);
X        = randn(n,p);
Y        = A*X;
W1       = orth(Y);
B        = W1'*A;
[W2,S,V] = svdecon(B);
U        = W1*W2;
k        = min(k,size(U,2));
U        = U(:,1:k);
S        = S(1:k,1:k);
V        = V(:,1:k);
