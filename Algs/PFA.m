%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Pareto-Following Algorithm and the 
% Natural Gradient.
% 
% Reference: S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli (2014)
% Policy gradient approaches for multi-objective sequential decision making
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

verboseOut = true; % to print output messages

domain = 'dam';
makeDet = 0; % if we want to consider stochastic or deterministic policies
[N_obj, init_pol, episodes, steps, gamma] = feval([domain '_settings']);
N_params = length(init_pol.theta);

tolerance_step = 0.01; % tolerance on the norm of the gradient (stopping condition) during the optimization step
tolerance_corr = 0.1; % the same, but during the correction step
minS = 0.1; % min entropy of the policy (with Gaussian policies the (differential) entropy can be negative)
lrate_single = 4; % lrate during single-objective optimization phase
lrate_step = 2; % lrate during optimization step
lrate_corr = 4; % lrate during correction step
maxIter = 200; % max iterations per optimization
maxCorrection = 25; % max iterations during the correction

front_pol = [];

n_iterations = 0; % total number of iterations (policy evaluations)

%% Learn the last objective
while true

    n_iterations = n_iterations + 1;
    [ds, J_init, S] = collect_samples(domain, episodes, steps, init_pol);

    [nat_grad, stepsize_single] = eNACbase(init_pol,ds,gamma,N_obj,lrate_single);
    dev = norm(nat_grad);
    
    str = strtrim(sprintf('%.4f, ', J_init));
    str(end) = [];
    if verboseOut, fprintf('Norm: %.4f, S: %.2f, J: [ %s ]\n', dev, S, str); end
    
    if dev < tolerance_step || S < minS || n_iterations > maxIter
        break
    end
    init_pol = init_pol.update(stepsize_single * nat_grad);
    
end

inter_pol = init_pol;
str = strtrim(sprintf('%.4f, ', J_init));
str(end) = [];
if verboseOut, fprintf('Initial Pareto solution found: [ %s ], S: %.2f \n', str, S); end

%% Learn the remaining objectives
for obj = -[-(N_obj-1) : -1] % for all the remaining objectives ...

    front_J = evaluate_policies([front_pol; inter_pol], domain, makeDet);
    [front_J, front_pol] = pareto(front_J, [front_pol; inter_pol]);
    inter_pol = [];
    
    num_policy = numel(front_pol);
    
    %% Loop for i-th objective
    for i = 1 : num_policy % ... for all the Pareto-optimal solutions found so far ...

        current_pol = front_pol(i);
        current_pol = current_pol.randomize(10); % SEE README!
        current_iter = 0; % number of steps for the single-objective optimization

        if verboseOut, fprintf('\n\nOptimizing objective %d ...\n', obj); end
        
        %% Loop for j-th policy
        while true % ... perform policy gradient optimization

            n_iterations = n_iterations + 1;
            current_iter = current_iter + 1;
    
            [ds, current_J, S] = collect_samples(domain, episodes, steps, current_pol);

            [nat_grad_step, stepsize_step] = eNACbase(current_pol,ds,gamma,obj,lrate_step);
            dev = norm(nat_grad_step);
            
            if verboseOut, fprintf('%d / %d ) Iter: %d, Norm: %.4f, S: %.2f \n', i, num_policy, current_iter, dev, S); end
            
            if dev < tolerance_step || current_iter > maxIter % stopping conditions
                str = strtrim(sprintf('%.4f, ', current_J));
                str(end) = [];
                if verboseOut, fprintf('Objective %d optimized! [ %s ], S: %.2f \n-------------\n', obj, str, S); end
                break
            end
            
            if S < minS
                current_pol = current_pol.randomize(10); % SEE README!
                if verboseOut, fprintf('RANDOMIZING! \n' ); end
                continue
            end
            
            current_pol = current_pol.update(stepsize_step * nat_grad_step); % perform an optimization step
            inter_pol = [inter_pol; current_pol]; % save intermediate solution
            
            iter_correction = 0;
            
            while true % correction phase
                
                [ds, current_J, S] = collect_samples(domain, episodes, steps, current_pol);

                M = zeros(N_params,N_obj);
                for j = 1 : N_obj
                    nat_grad = eNACbase(current_pol,ds,gamma,j);
                    M(:,j) = nat_grad / max(norm(nat_grad),1e-8); % always normalize during the correction
                end
                pareto_dir = paretoDirection(N_obj, M); % minimal-norm Pareto-ascent direction
                dev = norm(pareto_dir);
                
                str = strtrim(sprintf('%.4f, ', current_J));
                str(end) = [];
                if verboseOut, fprintf('   Correction %d, Norm: %.4f, S: %.2f, J: [ %s ]\n', iter_correction, dev, S, str); end
                
                if dev < tolerance_corr || iter_correction > maxCorrection % if on the frontier
                    break
                end
                
                if S < minS % deterministic policy not on the frontier
                    if verboseOut, fprintf('RANDOMIZING! \n' ); end
                    current_pol = current_pol.randomize(10); % SEE README!
                    break
                end

                current_pol = current_pol.update(lrate_corr * pareto_dir); % move towards the frontier
                inter_pol = [inter_pol; current_pol]; % save intermediate solution

                n_iterations = n_iterations + 1;
                iter_correction = iter_correction + 1;
                
            end
            
        end
        
    end
    
end


%% Plot
f = evaluate_policies([front_pol; inter_pol], domain, makeDet);
[f, p] = pareto(f, [front_pol; inter_pol]);

figure; hold all
if N_obj == 2
    plot(f(:,1),f(:,2),'g+')
end

if N_obj == 3
    scatter3(f(:,1),f(:,2),f(:,3),'g+')
end

feval([domain '_moref'],1);