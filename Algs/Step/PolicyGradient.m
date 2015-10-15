clear all
domain = 'deep';
robj = 1;
[n_obj, pol, episodes, steps, gamma] = feval([domain '_settings']);
iter = 0;

tolerance = 0.01;
lrate = 1;


%% Learning
while true
    
    iter = iter + 1;
    [ds, J] = collect_samples(domain, episodes, steps, pol);
    S = pol.entropy(horzcat(ds.s));

%     [grad, stepsize] = FiniteDifference(pol,episodes,J,domain,robj,lrate);
%     [grad, stepsize] = GPOMDPbase(pol,ds,gamma,robj,lrate);
%     [grad, stepsize] = eREINFORCEbase(pol,ds,gamma,robj,lrate);
    [grad, stepsize] = eNACbase(pol,ds,gamma,robj,lrate);
    
    norm_g = norm(grad);
    
    str_obj = strtrim(sprintf('%.4f, ', J));
    str_obj(end) = [];
    fprintf('%d ) Entropy: %.2f, Norm: %.4f, J: [ %s ]\n', iter, S, norm_g, str_obj)
    
    pol = pol.update(grad * stepsize);
end
