function LQR = lqr_environment ( n )

LQR.e = 0.1;
LQR.g = 0.9;
LQR.A = eye(n);
LQR.B = eye(n);
LQR.E = eye(n);
LQR.S = zeros(n);
% LQR.Sigma = zeros(n);
LQR.Sigma = eye(n);
LQR.x0 = zeros(0);

for i = 1 : n
    LQR.x0 = [LQR.x0; 10];
end
% LQR.x0 = [-3; 11];

for i = 1 : n
    LQR.Q{i} = eye(n);
    LQR.R{i} = eye(n);
end

for i = 1 : n
    for j = 1 : n
        if i == j
            LQR.Q{i}(j,j) = 1-LQR.e;
            LQR.R{i}(j,j) = LQR.e;
        else
            LQR.Q{i}(j,j) = LQR.e;
            LQR.R{i}(j,j) = 1-LQR.e;
        end
    end
end

end