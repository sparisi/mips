clear

N = 15;
N_MAX = N * 10;
robj = 1;

dim = 15;
mu0 = zeros(dim,1);
sigma0 = 100 * eye(dim);
policy_high = GaussianConstantChol(dim, mu0, sigma0);

solver = MORE_Solver(0.9,0.99,-75,policy_high);
% solver = NES_Solver(0.1);
% solver = REPS_Solver(0.9);

f = @(x) -sum( 100*(x(2:end,:) - x(1:end-1,:).^2).^2 + (1 - x(1:end-1,:)).^2 , 1 ); % Rosenbrock
f = @(x) -10*dim -sum( x.^2 - 10*cos(2*pi*x) , 1 ); % Rastrigin
% M = rand(dim); f = @(x) -sum((M*x).*x) -mymvnrnd(0,1,N).*abs(sum((M*x).*x)); % Noisy Sphere Function

iter = 1;

%% Learning
while N * iter < 1e4

    ds.Theta = policy_high.drawAction(N);
    ds.J = f(ds.Theta);
    avgRew = mean(ds.J(robj,:),2);
    
    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(ds.J(robj,:),[],2),1,N_MAX);
        Theta = policy_high.drawAction(N_MAX);
    end
        
    % Enqueue the new samples and remove the old ones
    J = [ds.J(robj,:), J(:, 1:N_MAX-N)];
    Theta = [ds.Theta, Theta(:, 1:N_MAX-N)];

    % Perform an update step
    [policy_high, divKL] = solver.step(J(robj,:),Theta,policy_high);
    
    J_history(:,iter) = ds.J(robj,:);
    fprintf( 'Iter: %d, Avg Reward: %.4f, KL Div: %.2f, Entropy: %.4f \n', ...
        iter, avgRew(robj), divKL, policy_high.entropy);
    
    iter = iter + 1;
    
end

%%
plothistory(J_history)
