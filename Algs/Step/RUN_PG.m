iter = 1;
lrate = 0.1;

%%
while true
    
    [ds, J] = collect_samples(mdp, episodes_learn, steps_learn, policy);
    S = policy.entropy(horzcat(ds.s));

%     [grad, stepsize] = GPOMDPbase(policy,ds,gamma,lrate);
%     [grad, stepsize] = eREINFORCEbase(policy,ds,gamma,lrate);
    [grad, stepsize] = eNACbase(policy,ds,gamma,lrate);
%     [grad, stepsize] = NaturalPG('r',policy,ds,gamma,lrate);
    
    J = evaluate_policies(mdp, episodes_learn, steps_learn, policy.makeDeterministic);
    norm_g = norm(grad(:,robj));
    fprintf('%d) Entropy: %.2f \tNorm: %.2e \tJ: %s \n', ...
        iter, S, norm_g, num2str(J','%.4f, '))
    J_history(iter) = J(robj);
    
%     updateplot('Return',iter,J,1)
    
%     policy = policy.update(policy.theta + grad(:,robj) * stepsize(robj));
    policy = policy.update(policy.theta + grad(:,robj) / norm(grad(:,robj)));
    
    iter = iter + 1;

end
