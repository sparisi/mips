clear

N = 15;
N_MAX = N * 10;
robj = 1;

dim = 15;
mu0 = zeros(dim,1);
sigma0 = 100 * eye(dim);
sampling = GaussianConstantChol(dim, mu0, sigma0);

solver = MORE_Solver(0.9,0.99,-75,sampling); divStr = 'KL Div';
% solver = NES_Solver(0.1); divStr = 'Norm';
% solver = REPS_Solver(0.9); divStr = 'KL Div';

f = @(x)rosenbrock(x);
f = @(x)rastrigin(x);
% f = @(x)noisysphere(x);

iter = 1;

%% Learning
while N * iter < 1e4

    X_iter = sampling.drawAction(N);
    Y_iter = f(X_iter);
    avgY = mean(Y_iter,2);
    
    % First, fill the pool to maintain the samples distribution
    if iter == 1
        Y = repmat(min(Y_iter,[],2),1,N_MAX);
        X = sampling.drawAction(N_MAX);
    end
        
    % Enqueue the new samples and remove the old ones
    Y = [Y_iter, Y(:, 1:N_MAX-N)];
    X = [X_iter, X(:, 1:N_MAX-N)];

    % Perform an update step
    [sampling, div] = solver.step(Y,X,sampling);
    
    J_history(:,iter) = Y_iter;
    fprintf( ['Iter: %d, Avg Value: %.4f, ' divStr ': %.2f, Entropy: %.4f \n'], ...
        iter, avgY, div, sampling.entropy );
    
    if div < 0.1, break, end
    
    iter = iter + 1;
    
end

%%
plothistory(J_history)
