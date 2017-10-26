clear, clear global
close all

N = 25;
N_MAX = N * 5;
N_eval = 1000;
MAX_ITER = 1000;

dim = 30; f = @(x)zdt1(x); constrain = @(x)max(min(1,x),0);
truefront = load('zdt1_front.dat');

nobj = size(truefront,2);
utopia = max(truefront);
antiutopia = min(truefront); 

normalized = 1;

if normalized
    normalize = @(y)normalize_data(y,antiutopia,utopia);
    utopia = ones(1,nobj);
    antiutopia = zeros(1,nobj);
else
    normalize = @(y)y;
end

if nobj == 2
    hyperv = @(y,antiutopia) hypervolume2d(normalize(y), antiutopia, utopia);
else
    hyperv = @(y,antiutopia) hypervolume(normalize(y), antiutopia, utopia, 1e6);
end

mu0 = rand(dim,1);
sigma0 = 10 * eye(dim);

% metric = @(y) metric_nd(normalize(y)); % Non-dominance-based metric
metric = @(y) metric_hv(normalize(y), @(x)hyperv(x,min([normalize(y);antiutopia]))); % Hyperv-based with adaptive antiutopia
% metric = @(y) metric_hv(normalize(y), @(x)hyperv(x,antiutopia)); % Hyperv-based with fixed antiutopia
% metric = @(y) metric_hv(normalize(y), @(x)hyperv(x,2*antiutopia)); % Hyperv-based with relaxed antiutopia

sampling = GaussianConstantChol(dim, mu0, sigma0);
solver = MORE_Solver(0.9,0.99,-75,sampling); divStr = 'KL Div';

iter = 1;

%% Learning
while true
    % Learning samples
    X_iter = sampling.drawAction(N);
    X_iter = constrain(X_iter);
    Y_iter = metric(f(X_iter)')';
    avgY(iter) = mean(Y_iter,2);
    
    % Eval samples
    X_eval = sampling.drawAction(N_eval);
    X_eval = constrain(X_eval);
    Y_eval = metric(f(X_eval)')';
    avgY(iter) = mean(Y_eval,2);
    
    % First, fill the pool to maintain the samples distribution
    if iter == 1
        Y = repmat(min(Y_iter,[],2),1,N_MAX);
        X = sampling.drawAction(N_MAX);
        X = constrain(X);
    end
    
    % Enqueue the new samples and remove the old ones
    Y = [Y_iter, Y(:, 1:N_MAX-N)];
    X = [X_iter, X(:, 1:N_MAX-N)];
    
    % Perform an update step
    [sampling, div] = solver.step(Y,X,sampling);
    
    fprintf( ['Iter: %d, Avg Value: %.4f, ' divStr ': %.2f, Entropy: %.4f \n'], ...
        iter, avgY(iter), div, sampling.entropy );
    
    iter = iter + 1;
    
    if div < 0.1, fprintf(' - %d\n', iter), break, end
    if iter >= MAX_ITER, fprintf(' - MAX ITER REACHED!\n'), break, end
    
end

%% Eval
X_eval = sampling.drawAction(N_eval);
X_eval = constrain(X_eval);
Y_eval = f(X_eval);
[front, p] = pareto(Y_eval', X_eval');
fprintf('Hypervolume: %.4f\n', hyperv(front,antiutopia));

figure, hold all
MOMDP.plotfront(truefront,'o');
MOMDP.plotfront(front,'+');

figure, plot(avgY);
autolayout
