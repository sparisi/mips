% Simple random search computing a finite difference approximation along 
% the random direction.
%
% =========================================================================
% REFERENCE
% https://arxiv.org/pdf/1803.07055.pdf

N = 20; % Number of samples collected per iteration
if makeDet, policy = policy.makeDeterministic; end

theta = policy_high.mu;
lrate = 0.0005;
iter = 1;

%% Learning
while true

    % Draw samples and create new policies
    Theta = policy_high.drawAction(N);
    Policies_pos = policy.empty(0,N);
    Policies_neg = policy.empty(0,N);
    for i = 1 : N
        Policies_pos(i) = policy.update(theta + Theta(:,i));
        Policies_neg(i) = policy.update(theta - Theta(:,i));
    end
    
    % Evaluate policies
    J_pos = evaluate_policies(mdp, 1, steps_learn, Policies_pos);
    J_neg = evaluate_policies(mdp, 1, steps_learn, Policies_neg);
    
    % Finite difference gradient
    theta = theta + lrate * mean(bsxfun(@times, Theta, (J_pos-J_neg)), 2);

    % Reduce exploration
    policy_high = policy_high.update(policy_high.mu, policy_high.Sigma*0.995); 

    % Print info
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.update(theta));
    J_history(iter) = J;
    fprintf( 'Iter: %d, J: %.4f\n', iter, J );
    iter = iter + 1;
    
end

%%
show_simulation(mdp, policy.update(theta), 1000, 0.01)