function qfunction = lqr_qfunction( A, B, Q, R, K, Sigma, x, u, g )

P = lqr_pmatrix(A, B, Q, R, K, g);

if g == 1
    qfunction = x'*Q*x + u'*R*u + g * (A*x + B*u)' * P * (A*x + B*u) - ...
        trace( Sigma * (R + g*B'*P*B) );
else
    qfunction = x'*Q*x + u'*R*u + g * (A*x + B*u)' * P * (A*x + B*u) + ...
        (g/(1-g)) * trace( Sigma * (R + g*B'*P*B) );
end

end