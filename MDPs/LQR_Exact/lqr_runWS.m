%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the weighted sum approach.
% The policy is Gaussian with linear mean and constant variance.
% Only the mean is learned.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

step = 10; % Inverse of the stepsize in the weights interval
N_obj = 3;

LQR = lqr_init(N_obj);
g = LQR.g;
A = LQR.A;
B = LQR.B;
x0 = LQR.x0;
Sigma = LQR.Sigma;

% Generate all combinations of weights for the objectives
W = convexWeights(N_obj, step);
N_sol = size(W,1);
FrontJ = zeros(N_sol, N_obj);
FrontK = cell(N_sol,1);
FrontSigma = cell(N_sol,1);

parfor k = 1 : N_sol

    Q = zeros(N_obj);
    R = zeros(N_obj);
    for i = 1 : N_obj
        Q = Q + W(k,i) * LQR.Q{i};
        R = R + W(k,i) * LQR.R{i};
    end
    
    % Compute the optimal controller in closed form
    P = eye(size(Q,1));
    for i = 1 : 100
        K = -g*inv(R+g*(B'*P*B))*B'*P*A;
        P = Q + g*A'*P*A + g*K'*B'*P*A + g*A'*P*B*K + g*K'*B'*P*B*K + K'*R*K;
    end
    K = -g*inv(R+g*B'*P*B)*B'*P*A;
    
    % Evaluate objectives
    J = zeros(1,N_obj);
    for i = 1 : N_obj
        J(i) = lqr_return(A,B,LQR.Q{i},LQR.R{i},K,Sigma,x0,g);
    end
    FrontJ(k,:) = J;
    FrontK{k} = K;
    FrontSigma{k} = Sigma;
    
end
    
toc;

%% Plot
figure; hold all

if N_obj == 2
    plot(FrontJ(:,1),FrontJ(:,2),'g+')
    xlabel('J_1'); ylabel('J_2');
end

if N_obj == 3
    scatter3(FrontJ(:,1),FrontJ(:,2),FrontJ(:,3),'g+')
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
end

legend(['\epsilon = ' num2str(LQR.e) ', \gamma = ' num2str(LQR.g) ...
    ', \Sigma = I, x_0 = ' mat2str(LQR.x0)],'Location','NorthOutside')
