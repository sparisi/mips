clear all

rng(10)

N = 300;
N_MAX = N * 1;
N_eval = 1;
robj = 1;

dim = 15;
mu0 = zeros(dim,1);
sigma0 = 100 * eye(dim);
sampling = GaussianConstantChol(dim, mu0, sigma0);
% sampling = GmmConstant(mu0,sigma0,4);

solver = MORE_Solver(0.9,0.99,-75000,sampling); divStr = 'KL';
% solver = MORE2_Solver(0.9,sampling); divStr = 'KL';
% solver = NES_Solver(0.1); divStr = 'Grad Norm';
% solver = REPSep_Solver(0.9); divStr = 'KL (Weights)';
solver = REPSep_constrained_Solver(0.9,1); divStr = 'KL (Weights)';

f = @(x)rosenbrock(x);
% f = @(x)rastrigin(x);
% f = @(x)noisysphere(x); N_eval = 1000;

iter = 1;

%% Learning
while iter < 150

    % Learning samples
    X_iter = sampling.drawAction(N);
    Y_iter = f(X_iter);

    % Eval samples
    X_eval = sampling.makeDeterministic.drawAction(N_eval);
    Y_eval = f(X_eval);
    J_history(:,iter) = Y_eval;

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        Y = repmat(min(Y_iter,[],2),1,N_MAX);
        X = sampling.drawAction(N_MAX);
    end
        
    % Enqueue the new samples and remove the old ones
    Y = [Y_iter, Y(:, 1:N_MAX-N)];
    X = [X_iter, X(:, 1:N_MAX-N)];

    % Perform an update step
    sampling_old = sampling;
    [sampling, div] = solver.step(Y,X,sampling);
    
    fprintf( ['Iter: %d, Avg Value: %.4f, ' divStr ': %.4f, Entropy: %.4f'], ...
        iter, mean(Y_eval), div, sampling.entropy(X) );
    if isa(sampling,'Gaussian')
        fprintf(', KL: %.4f', kl_mvn(sampling, sampling_old));
    end
    fprintf('\n');
    
    iter = iter + 1;
    
end

%%
if N_eval == 1
    plot(J_history)
else
    plothistory(J_history)
end
