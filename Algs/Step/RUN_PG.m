iter = 1;
lrate = 0.1;

%%
while true
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    S = policy.entropy([ds.s]);

%     [grad, stepsize] = GPOMDPbase(policy,ds,mdp.gamma,lrate);
%     [grad, stepsize] = eREINFORCEbase(policy,ds,mdp.gamma,lrate);
    [grad, stepsize] = eNACbase(policy,ds,mdp.gamma,lrate);
    
%     J = evaluate_policies(mdp, episodes_eval, steps_eval, policy.makeDeterministic);
    J = evaluate_policies(mdp, episodes_eval, steps_eval, policy);
    
    norm_g = norm(grad(:,robj));
    fprintf('%d) Entropy: %.2f \tNorm: %.2e \tJ: %s \n', ...
        iter, S, norm_g, num2str(J','%.4f, '))
    J_history(iter) = J(robj);
    
%     updateplot('Return',iter,J,1)
    
    policy = policy.update(policy.theta + grad(:,robj) * stepsize(robj));
%     policy = policy.update(policy.theta + grad(:,robj) / norm(grad(:,robj)));
    
    iter = iter + 1;

end
