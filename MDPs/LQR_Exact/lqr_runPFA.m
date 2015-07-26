%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Pareto-Following Algorithm and the 
% Natural Gradient.
% The policy is Gaussian with linear mean and constant variance.
% Only the mean is learned.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;

N = 3;
LQR = lqr_init(N);
g = LQR.g;
A = LQR.A;
B = LQR.B;
Q = LQR.Q;
R = LQR.R;
e = LQR.e;
x0 = LQR.x0;
Sigma = LQR.Sigma;

% Initial policy
initK = -0.5 * eye(N);

tolerance_step = 0.1; % Tolerance on the norm of the gradient (stopping condition) for the optimization step
tolerance_corr = 0.1; % The same, but for the correction step
lrate_single = 0.1; % Lrate for the single-objective optimization phase
lrate_step = 0.05; % Lrate for the optimization step
lrate_corr = 0.1; % Lrate for the correction step
total_iter = 0; % Total number of iterations (policy evaluations)


%% Learn the last objective
while true

    total_iter = total_iter + 1;
    nat_grad_init = calcNatGradient(A,B,Q{N},R{N},initK,Sigma,g);
    dev = norm(diag(nat_grad_init));
    if dev < tolerance_step
        break
    end
    initK = initK - lrate_single * nat_grad_init;
    
end

initJ = zeros(1,N); % Save the first solution, i.e. one extreme point of the frontier
for i = 1 : N
    initJ(i) = lqr_return(A,B,Q{i},R{i},initK,Sigma,x0,g);
end
FrontK = {initK}; % store Pareto-frontier policies
FrontJ = initJ; % store Pareto-frontier
interK = {}; % store not-Pareto-optimal policies found during correction steps
interJ = []; % store not-Pareto-optimal solutions


%% Learn the remaining objectives
for obj = 1 : N-1 % for all the remaining objectives ...

    current_frontK = FrontK;
    
    for i = 1 : numel(current_frontK) % ... for all the Pareto-optimal solutions found so far ...

        current_K = current_frontK{i};
        current_iter = 0; % number of steps for the single-objective optimization

        while true % ... perform policy gradient optimization

            total_iter = total_iter + 1;
            current_iter = current_iter + 1;
    
%             lrate_step = 5*1e-4*current_iter; % adaptive lrate to obtain a uniform frontier

            nat_grad_step = calcNatGradient(A,B,Q{obj},R{obj},current_K,Sigma,g);
            dev = norm(diag(nat_grad_step));
            if dev < tolerance_step % break if the objective has been learned
                current_J = zeros(1,N);
                for j = 1 : N
                    current_J(j) = lqr_return(A,B,Q{j},R{j},current_K,Sigma,x0,g);
                end
                FrontK = [FrontK; current_K]; % save Pareto solution
                FrontJ = [FrontJ; current_J];
                break
            end
            
            current_K = current_K - lrate_step * nat_grad_step; % perform an optimization step
            
            while true % correction phase
                
                current_J = zeros(1,N);
                for j = 1 : N
                    current_J(j) = lqr_return(A,B,Q{j},R{j},current_K,Sigma,x0,g);
                end
                
                M = zeros(N^2,N); % jacobian
                for j = 1 : N
                    nat_grad_corr = calcNatGradient(A,B,Q{j},R{j},current_K,Sigma,g);
                    M(:,j) = nat_grad_corr(:);
                end
                options = optimset('Display','off');
                lambda = quadprog(M'*M, zeros(N,1), [], [], ones(1,N), ...
                    1, zeros(1,N), [], ones(N,1)/N, options);
                dir = M*lambda; % minimal-norm common distance
                dev = norm(dir);
                if dev < tolerance_corr % if on the frontier
                    FrontK = [FrontK; current_K]; % save Pareto solution
                    FrontJ = [FrontJ; current_J];
                    break
                end

                interK = [interK; current_K]; % save intermediate solution
                interJ = [interJ; current_J];

                nat_grad_corr = vec2mat(dir,N);
                current_K = current_K - lrate_corr * nat_grad_corr; % move towards the frontier

                total_iter = total_iter + 1;
                
            end
            
        end
        
    end
    
end

toc;

%% Plot
figure; hold all

if N == 2
    plot(FrontJ(:,1),FrontJ(:,2),'g+')
    xlabel('J_1'); ylabel('J_2');
end

if N == 3
    scatter3(FrontJ(:,1),FrontJ(:,2),FrontJ(:,3),'g+')
    xlabel('J_1'); ylabel('J_2'); zlabel('J_3');
end

legend(['\epsilon = ' num2str(e) ', \gamma = ' num2str(g) ...
    ', \Sigma = I, x_0 = ' mat2str(x0)],'Location','NorthOutside')
