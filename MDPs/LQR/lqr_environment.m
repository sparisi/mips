function LQR = lqr_environment ( n )

LQR.e = 0.1;
LQR.A = eye(n);
LQR.B = eye(n);
LQR.x0 = 10*ones(n,1);

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