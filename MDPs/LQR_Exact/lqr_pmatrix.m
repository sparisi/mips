function P = lqr_pmatrix( A, B, Q, R, K, g )
% Riccati matrix.

I = eye(size(K,1));

P = (Q + K * R * K) / (I - g * (I + 2 * K + K^2)); % only if A = B = I

% tolerance = 0.0001;
% converged = false;
% P = I;
% Pnew = Q + g*A'*P*A + g*K'*B'*P*A + g*A'*P*B*K + g*K'*B'*P*B*K + K'*R*K;
% while ~converged
%     P = Pnew;
%     Pnew = Q + g*A'*P*A + g*K'*B'*P*A + g*A'*P*B*K + g*K'*B'*P*B*K + K'*R*K;
%     converged = max(abs(P(:)-Pnew(:))) < tolerance;
% end

end

