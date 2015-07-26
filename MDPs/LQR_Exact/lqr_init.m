function LQR = lqr_init ( n )
% Initializes a LQR with N conflictual objectives.

LQR.dim = n;

if n == 1
    
    LQR.g = 0.95;
    LQR.A = 1;
    LQR.B = 1;
    LQR.Q{1} = 1;
    LQR.R{1} = 1;
    LQR.E = 1;
    LQR.S = 0;
    LQR.Sigma = 0.01;
    LQR.x0 = 10;
    return
    
end

LQR.e = 0.1;
LQR.g = 0.9;
LQR.A = eye(n);
LQR.B = eye(n);
LQR.E = eye(n);
LQR.S = zeros(n);
LQR.Sigma = eye(n);
LQR.x0 = 10 * ones(n,1);

for i = 1 : n
    LQR.Q{i} = eye(n) * LQR.e;
    LQR.R{i} = eye(n) * (1-LQR.e);
    LQR.Q{i}(i,i) = 1-LQR.e;
    LQR.R{i}(i,i) = LQR.e;
end

end