function vfunction = lqr_vfunction( A, B, Q, R, K, Sigma, x, g )

P = lqr_pmatrix(A,B,Q,R,K,g);

if g == 1
    vfunction = x'*P*x;
else
    vfunction = x'*P*x + (1/(1-g)) * trace( Sigma * (R + g*B'*P*B) );
end

end