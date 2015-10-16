function J = lqr_return( A, B, Q, R, K, Sigma, x0, g )

P = lqr_pmatrix(A,B,Q,R,K,g);

if g == 1 % average return
    J = trace(Sigma*(R+B'*P*B));
else % discounted return
    J = x0'*P*x0 + (1/(1-g))*trace(Sigma*(R+g*B'*P*B));
end

end