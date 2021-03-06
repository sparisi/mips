clear all

rng(91)

N = 15;
N_MAX = N * 20;
N_eval = 1000;

dim_ctx = 3;
dim_action = 1;

lb = -10*ones(dim_ctx,1);
ub = 10*ones(dim_ctx,1);
ctx = @(varargin) myunifrnd(lb, ub, varargin{:});

a0 = 5 * rand(dim_action,dim_ctx+1);
sigma0 = 10000 * eye(dim_action);
bfsP = @(varargin) basis_poly(1, dim_ctx, 0, varargin{:});
sampling = GaussianLinearChol(bfsP, dim_action, a0, sigma0);

bfsV = @(varargin)basis_poly(2, dim_ctx, 0, varargin{:});
% bfsV = @(varargin)basis_krbf(3, [lb ub], 0, varargin{:});
solver = CREPS_Solver(0.9,bfsV);

f = @(action,ctx)quadcost(action,ctx);

iter = 1;

%% Learning
MAXITER = 100;
while iter < MAXITER

    % Learning samples
    C_iter = ctx(N);
    A_iter = sampling.drawAction(C_iter);
    PhiP_iter = sampling.get_basis(C_iter);
    PhiV_iter = bfsV(C_iter);
    R_iter = f(A_iter,C_iter);
    avgY = mean(R_iter,2);
    
    % Eval samples
    C_eval = ctx(N_eval);
    A_eval = sampling.makeDeterministic.drawAction(C_eval);
    R_eval = f(A_eval,C_eval);
    avgY = mean(R_eval,2);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        C = ctx(N_MAX);
        R = repmat(min(R_iter,[],2),1,N_MAX);
        PhiP = sampling.get_basis(C);
        PhiV = bfsV(C);
        A = sampling.drawAction(C);
    end
        
    % Enqueue the new samples and remove the old ones
    R = [R_iter, R(:, 1:N_MAX-N)];
    A = [A_iter, A(:, 1:N_MAX-N)];
    C = [C_iter, C(:, 1:N_MAX-N)];
    PhiP = [PhiP_iter, PhiP(:, 1:N_MAX-N)];
    PhiV = [PhiV_iter, PhiV(:, 1:N_MAX-N)];

    % Perform an update step
    [d, divKL] = solver.optimize(R,PhiV);
    sampling = sampling.weightedMLUpdate(d, A, PhiP);
    
    J_history(:,iter) = R_eval;
    fprintf( ['Iter: %d, Avg Value: %.4f, KL: %.2f, Entropy: %.4f \n'], ...
        iter, avgY, divKL, sampling.entropy );
        
    iter = iter + 1;

    solver.plotV(lb, ub)
    
end

J_history(:,end+1:MAXITER) = avgY;
