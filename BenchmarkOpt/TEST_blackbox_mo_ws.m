clear
close all

N = 100;
N_MAX = N * 1;
N_eval = 1;
MAX_ITER = 1000;

dim = 30; f = @(x)zdt3(x); constrain = @(x)max(min(1,x),0);
truefront = load('zdt3_front.dat');

nobj = size(truefront,2);
utopia = max(truefront);
antiutopia = min(truefront); 

normalized = 1;

if normalized
    normalize = @(y)normalize_data(y',antiutopia,utopia);
    utopia = ones(1,nobj);
    antiutopia = zeros(1,nobj);
else
    normalize = @(y)y';
end

mu0 = zeros(dim,1);
sigma0 = 2 * eye(dim);

step = 150;

% metric = @(y,w)metric_ws(normalize(y),w)'; % Linear scalarization function
% metric = @(y,w)metric_cheby(normalize(y),w,utopia)'; % Chebychev scalarization
% W = convexcomb(nobj, step);

metric = @(y,w)metric_localutopia(normalize(y),w,utopia,antiutopia)'; % Local-utopia distance
W = linspacesim(antiutopia,utopia,step)'; % Weights in the simplex
W = -linspacesim(-utopia,-(antiutopia-abs(antiutopia))*0.2,step)'; % Weights in the 'upper' simplex

%% Loop
for i = 1 : size(W,1)

    fprintf( '\n %d / %d', i, size(W,1) );

    sampling = GaussianConstantChol(dim, mu0, sigma0);

    solver = MORE_Solver(0.9,0.99,-75,sampling); divStr = 'KL Div';
%     solver = NES_Solver(0.1); divStr = 'Norm';
    solver = REPSep_Solver(0.9); divStr = 'KL Div';

    iter = 1;

    %% Learning
    while true

        % Learning samples
        X_iter = sampling.drawAction(N);
        X_iter = constrain(X_iter);
        Y_iter = metric(f(X_iter),W(i,:));
        avgY = mean(Y_iter,2);

        % Eval samples
        X_eval = sampling.makeDeterministic.drawAction(N_eval);
        X_eval = constrain(X_eval);
        Y_eval = f(X_eval);
        avgY = mean(Y_eval,2);

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

        % fprintf( ['%d / %d) Iter: %d, Avg Value: %.4f, ' divStr ': %.2f, Entropy: %.4f \n'], ...
        %     i, size(W,1), iter, avgY, div, sampling.entropy );

        iter = iter + 1;

        if div < 0.1, fprintf(' - %d\n', iter), break, end
        if iter >= MAX_ITER, fprintf(' - MAX ITER REACHED!\n'), break, end

    end

    front_pol(i) = sampling;

end

%% Eval and plot
for i = 1 : length(front_pol)
    front(i,:) = f(constrain(front_pol(i).makeDeterministic.drawAction));
end
f = pareto(front);
figure, hold all
MOMDP.plotfront(truefront,'o');
MOMDP.plotfront(front,'+');
