clear all

rng(1)

maxiter = 100;
N = 100;
N_MAX = N * 1;
N_eval = 1000;

dim_ctx = 10;
dim_action = 10;
% bfs = @(varargin) basis_fourier(50, dim_ctx, 0.2, 0, varargin{:});
bfs = @(varargin) basis_poly(1, dim_ctx, 0, varargin{:});
dim_bfs = bfs();

ctx_lb = 1*ones(dim_ctx,1);
ctx_ub = 2*ones(dim_ctx,1);
ctx = @(varargin) myunifrnd(ctx_lb, ctx_ub, varargin{:});

a0 = 1 * rand(dim_action,dim_bfs+1);
sigma0 = 10 * eye(dim_action);
sampling = GaussianLinearChol(@(varargin)basis_poly(1,dim_bfs,0,varargin{:}), dim_action, a0, sigma0);

solver = CMORE_Solver(0.9,0.99,-75,sampling);
solver.lambda_l1 = 0;
solver.lambda_l2 = 0.01;
solver.lambda_nn = 0.01;
solver.alg_name = 'fista';

G = rand(dim_bfs,dim_action);
f = @(action,ctx)rosenbrock_ctx(action,ctx,G);
f = @(action,ctx)quadcost(action,ctx); % For the quadratic cost you need to use dim_ctx = dim_action and poly(1) features

iter = 1;

%% Learning
while iter < maxiter

    % Learning samples
    C_iter = bfs(ctx(N));
    A_iter = sampling.drawAction(C_iter);
    R_iter = f(A_iter,C_iter);
    avgY = mean(R_iter,2);
    
    % Eval samples
    C_eval = bfs(ctx(N_eval));
    A_eval = sampling.makeDeterministic.drawAction(C_eval);
    R_eval = f(A_eval,C_eval);
    avgY = mean(R_eval,2);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        C = bfs(ctx(N_MAX));
        R = repmat(min(R_iter,[],2),1,N_MAX);
        A = sampling.drawAction(C);
    end
        
    % Enqueue the new samples and remove the old ones
    R = [R_iter, R(:, 1:N_MAX-N)];
    A = [A_iter, A(:, 1:N_MAX-N)];
    C = [C_iter, C(:, 1:N_MAX-N)];

    % Perform an update step
    [sampling, divKL] = solver.step(R,A,C,sampling);
    
    J_history(:,iter) = R_eval;
    fprintf( ['Iter: %d, Avg Value: %.4f, KL: %.2f, Entropy: %.4f \n'], ...
        iter, avgY, divKL, sampling.entropy );
        
    iter = iter + 1;
    
    solver.plotR(ctx_lb, ctx_ub, ctx_lb, ctx_ub)
    
end

%%
J_history(:,end+1:maxiter) = avgY;
if N_eval == 1
    plot(J_history)
else
    plothistory(J_history)
end
