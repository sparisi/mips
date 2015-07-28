clear all
domain = 'deep';
robj = 1;
[n_obj, pol, episodes, steps, gamma] = feval([domain '_settings']);
iter = 0;
theta = [];

tolerance = 0.001;
minS = -20; % with Gaussian policies the (differential) entropy can be negative
lrate = 0.1;


%% Learning
while true
    
    iter = iter + 1;
    [ds, J, S] = collect_samples(domain, episodes, steps, pol);

%     [grad, stepsize] = FiniteDifference(pol,episodes,J,domain,robj,lrate);
%     [grad, stepsize] = GPOMDPbase(pol,ds,gamma,robj,lrate);
%     [grad, stepsize] = eREINFORCEbase(pol,ds,gamma,robj,lrate);
    [grad, stepsize] = eNACbase(pol,ds,gamma,robj,lrate);
    
    theta = [theta; pol.theta'];
    norm_g = norm(grad);
    
    str_obj = strtrim(sprintf('%.4f, ', J));
    str_obj(end) = [];
    fprintf('%d ) Entropy = %.2f, Norm = %.4f, J = [ %s ]\n', iter, S, norm_g, str_obj)
    
    if norm_g < tolerance || S < minS
        break
    end
    
    pol = pol.update(grad * stepsize);

end