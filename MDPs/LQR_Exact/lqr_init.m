function LQR = lqr_init (n_obj)
% Reference: Parisi et al, Studio e analisi di algoritmi di apprendimento 
% per rinforzo policy gradient per la risoluzione di problemi decisionali 
% multiobiettivo (2014)

LQR.dim = n_obj;

if n_obj == 1
    
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
LQR.A = eye(n_obj);
LQR.B = eye(n_obj);
LQR.E = eye(n_obj);
LQR.S = zeros(n_obj);
LQR.Sigma = eye(n_obj);
LQR.x0 = 10 * ones(n_obj,1);

for i = 1 : n_obj
    LQR.Q{i} = eye(n_obj) * LQR.e;
    LQR.R{i} = eye(n_obj) * (1-LQR.e);
    LQR.Q{i}(i,i) = 1-LQR.e;
    LQR.R{i}(i,i) = LQR.e;
end

end