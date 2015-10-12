clear all

domain = 'deep';
robj = 1;
[n_obj, pol_high] = settings_episodic(domain,1);

N = 20;
N_MAX = N * 10;
lrate = 0.1;

J = zeros(N_MAX,n_obj);
Theta = zeros(pol_high.dim,N_MAX);

J_history = [];
iter = 0;

%% Learning
while iter < 200

    iter = iter + 1;

    [J_iter, Theta_iter] = collect_episodes(domain, N, pol_high);

    % First, fill the pool to maintain the samples distribution
    if iter == 1
        J = repmat(min(J_iter),N_MAX,1);
        for k = 1 : N_MAX
            Theta(:,k) = pol_high.drawAction;
        end
    end
    
    % Enqueue the new samples and remove the old ones
    J = [J_iter; J(1:N_MAX-N,:)];
    Theta = [Theta_iter, Theta(:, 1:N_MAX-N)];
    
    [grad, stepsize] = NESbase(pol_high, J(:,robj), Theta, lrate);
%     [grad, stepsize] = PGPEbase(pol_high, J(:,robj), Theta, lrate);

    avgRew = mean(J_iter(:,robj));
    J_history = [J_history, J_iter(:,robj)];
    fprintf( 'Iter: %d, Avg Reward: %.4f, Norm: %.2f, Entropy: %.3f\n', ...
        iter, avgRew, norm(grad), pol_high.entropy );

    % Ending condition
    if norm(grad) < 0.0001
%         break
    else
        pol_high = pol_high.update(grad * stepsize);
    end
    
end