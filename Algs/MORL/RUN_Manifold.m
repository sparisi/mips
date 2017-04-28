% REFERENCE
% S Parisi, M Pirotta, J Peters
% Manifold-based Multi-objective Policy Search with Sample Reuse (2016)

N = 10;
N_MAX = N * 5;
% solver = REPS_Solver(1); distance = 'KL Div';
solver = NES_Solver(0.1); distance = 'Norm';
% solver = MORE_Solver(0.9,0.99,-75,policy_high);

if makeDet, policy = policy.makeDeterministic; end

utopia = mdp.utopia;
antiutopia = mdp.antiutopia;

if dreward == 2
    hyperv = @(J,antiutopia) hypervolume2d(J, antiutopia, utopia);
else
    hyperv = @(J,antiutopia) hypervolume(J, antiutopia, utopia, 1e6);
end

% metric = @(J) metric_nd(J); % Non-dominance-based metric
metric = @(J) metric_hv(J, @(x)hyperv(x,min([J;antiutopia]))); % Hyperv-based with adaptive antiutopia
metric = @(J) metric_hv(J, @(x)hyperv(x,antiutopia)); % Hyperv-based with fixed antiutopia
% metric = @(J) metric_hv(J, @(x)hyperv(x,2*antiutopia)); % Hyperv-based with relaxed antiutopia
% metric = @(J) mexMetric_hv(J, antiutopia*2, utopia, 1e6); % Used for Dam3 and LQR5
% metric = @(J) mexMetric_hv(J, antiutopia, utopia, 1e6);

iter = 1;
MAXEPISODES = 160*1e3;
MAXITER = MAXEPISODES / N / episodes_learn

%% Plotting Setup
showfront = 0;
if showfront
    subplot(1,2,1)
    mdp.plotfront(mdp.truefront,'o');
    hold all
    frontLine1 = mdp.plotfront(antiutopia,'+');
    title('Objective Space')
    
    subplot(1,2,2)
    mdp.plotfront(mdp.truefront,'o','DisplayName','True frontier');
    hold all
    frontLine2 = mdp.plotfront(antiutopia,'+','DisplayName','Approximate frontier');
    limits = [mdp.antiutopia; mdp.utopia];
    axis(limits(:)');
    legend show
    title('Magnification')
end
    
%% Learning
while iter <= MAXITER

    % Draw N policies and evaluate them
    Theta_iter = policy_high.drawAction(N);
    p = repmat(policy,1,N);
    for i = 1 : N
        p(i) = p(i).update(Theta_iter(:,i));
    end
    J_iter = evaluate_policies(mdp, episodes_learn, steps_learn, p);
    
    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter,[],2),1,N_MAX);
        Theta = policy_high.drawAction(N_MAX);
        Policies = repmat(policy_high,1,N_MAX);
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter, J(:, 1:N_MAX-N)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];
    Policies = [repmat(policy_high,1,N), Policies(:, 1:N_MAX-N)];
    
    % Compute IS weights
    W = mixtureIS(policy_high, Policies, N, Theta);
%     W = ones(1,N_MAX);

    % Scalarize samples
    fitness = metric(J')';
    
    % Perform an update step
    [policy_high, div] = solver.step(fitness, Theta, policy_high, W);

    if div < 1e-6, continue, end
    
    hv = hyperv(pareto(J'),antiutopia);
    fprintf( 'Iter: %d, Hyperv: %.4f, %s: %.2f, Entropy: %.4f\n', ...
        iter, hv, distance, div, policy_high.entropy(Theta_iter) );
    hyperv_history(iter) = hv;
    
    iter = iter + 1;
    
    if showfront
        frontLine1.XData = J(1,:);
        frontLine1.YData = J(2,:);
        try frontLine1.ZData = J(3,:); catch, end
        frontLine2.XData = J(1,:);
        frontLine2.YData = J(2,:);
        try frontLine2.ZData = J(3,:); catch, end
        drawnow limitrate
    end
    
end

%% Eval
N_EVAL = 200;
Theta_eval = policy_high.drawAction(N_EVAL);

for i = 1 : N_EVAL
    pol_eval(i) = policy.update(Theta_eval(:,i));
end

f_eval = evaluate_policies(mdp, episodes_eval, steps_eval, pol_eval)';
[f, p] = pareto(f_eval, pol_eval);
fprintf('Hypervolume: %.4f\n', hyperv(f,antiutopia));

figure, hold all
mdp.plotfront(mdp.truefront,'o','DisplayName','True frontier');
mdp.plotfront(f,'+','DisplayName','Approximate frontier');
legend show

figure, plot((0:iter-2)*N*episodes_learn,hyperv_history);
autolayout
