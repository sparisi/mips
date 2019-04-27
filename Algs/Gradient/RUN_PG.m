iter = 1;
lrate = 0.01;

%%
while true
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    S = policy.entropy([ds.s]);

%     [grad, stepsize] = GPOMDPbase(policy,ds,mdp.gamma,lrate);
%     [grad, stepsize] = eREINFORCEbase(policy,ds,mdp.gamma,lrate);
%     [grad, stepsize] = eNACbase(policy,ds,mdp.gamma,lrate);
%     [grad, stepsize] = REINFORCE_C(policy,ds,mdp.gamma,lrate);
    [grad, stepsize] = NPG_C(policy,ds,mdp.gamma,lrate);
    
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
%     J = evaluate_policies(mdp, episodes_eval, steps_eval, policy);
    
    norm_g = norm(grad(:,robj));
    fprintf('%d) Entropy: %.2f \tNorm: %.2e \tJ: %s', ...
        iter, S, norm_g, num2str(J','%.4f, '))
    J_history(iter) = J(robj);
    
%     updateplot('Return',iter,J,1)
    
    policy_old = policy;
    policy = policy.update(policy.theta + grad(:,robj) * stepsize(robj));
%     policy = policy.update(policy.theta + grad(:,robj) / norm(grad(:,robj)));
    if isa(policy,'Gaussian')
        fprintf(' \tKL: %.4f', kl_mvn2(policy, policy_old, policy.basis([ds.s])));
    end
    fprintf('\n')
    iter = iter + 1;

end
