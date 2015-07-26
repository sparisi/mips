function advantage = lqr_advantage( A, B, Q, R, K, Sigma, x, u, g )

advantage = lqr_qfunction(A,B,Q,R,K,Sigma,x,u,g) - ...
    lqr_vfunction(A,B,Q,R,K,Sigma,x,g);

end