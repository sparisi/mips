%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Radial Algorithm and Natural 
% Evolution Strategies.
% 
% Reference: S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli (2014)
% Policy gradient approaches for multi-objective sequential decision making
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

verboseOut = true; % to print output messages

domain = 'dam';
makeDet = 0; % evaluate only deterministic policies?
[n_obj, init_pol] = settings_episodic(domain,1);
n_params = size(init_pol.theta,1);

N_episodes = 50;
simstep = 50; % density of the directions in the simplex
tolerance = 0.1; % tolerance for the norm of the gradient
maxIter = 200; % max number of policy gradient steps in the same direction
minS = 1.5; % min entropy of the policy (with Gaussian policies the (differential) entropy can be negative)
lrate = 1;

% generate all combinations of weights for the directions in the simplex
W = convexWeights(n_obj, simstep);
N_sol = size(W,1);
front_pol = cell(N_sol,1);
inter_pol = [];

% initial solution
[J_init, Theta_init] = collect_episodes(domain, N_episodes, init_pol);
M_init = zeros(n_params,n_obj);
for j = 1 : n_obj
    [nat_grad, stepsize] = NESbase(init_pol, J_init(:,j), Theta_init, lrate);
    M_init(:,j) = nat_grad * stepsize;
end

J_init = mean(J_init,1);
str_init = strtrim(sprintf('%.4f, ', J_init));
str_init(end) = [];
if verboseOut, fprintf('Starting from solution: [ %s ] \n', str_init); end

n_iterations = 0; % total number of iterations (policy evaluations)

%% For all the directions in the simplex
parfor k = 1 : N_sol
    
    fixedLambda = W(k,:)';
    newDir = M_init * fixedLambda;
    curr_pol = init_pol.update(newDir);

    iter = 1;
    str_dir = strtrim(sprintf('%.3f, ', fixedLambda));
    str_dir(end) = [];

    %% Policy gradient learning in one fixed direction
    while true
        
        n_iterations = n_iterations + 1;

        [J, Theta] = collect_episodes(domain, N_episodes, curr_pol);
        S = curr_pol.entropy;

        M = zeros(n_params,n_obj); % jacobian
        Mn = zeros(n_params,n_obj); % jacobian with normalized gradients
        for j = 1 : n_obj
            [nat_grad, stepsize] = NESbase(curr_pol, J(:,j), Theta, lrate);
            M(:,j) = nat_grad * stepsize;
            Mn(:,j) = nat_grad / max(norm(nat_grad), 1e-8); % to avoid numerical problems
        end
        
        dir = M * fixedLambda;
        dev = norm(dir);

        str_obj = strtrim(sprintf('%.4f, ', mean(J,1)));
        str_obj(end) = [];
        if verboseOut, fprintf('[ %s ] || Iter %d ) Norm: %.4f, S: %.2f, J: [ %s ] \n', ...
            str_dir, iter, dev, S, str_obj); end

        if dev < tolerance
            if verboseOut, fprintf('Cannot proceed any further in this direction!\n'); end
            front_pol{k} = curr_pol;
            break;
        end

%         % for the min-norm Pareto-ascent direction, always use Mn
%         dirPareto = paretoDirection(N_obj, Mn);
%         devPareto = norm(dirPareto);
% 
%         if devPareto < tolerance
%             if verboseOut, fprintf('Pareto front reached!\n'); end
%             front_pol{k} = curr_pol;
%             break;
%         end
        
        if S < minS
            if verboseOut, fprintf('Deterministic policy found!\n'); end
            front_pol{k} = curr_pol;
            break;
        end
        
        if iter > maxIter
            if verboseOut, fprintf('Iteration limit reached!\n'); end
            front_pol{k} = curr_pol;
            break;
        end
        
%         inter_pol = [inter_pol; curr_pol];
        iter = iter + 1;

        curr_pol = curr_pol.update(dir);

    end
    
end

front_pol = vertcat(front_pol{:});


%% Plot
fr = evaluate_policies_episodic(front_pol, domain, makeDet);
[f, p] = pareto(fr, front_pol);

figure; hold all
if n_obj == 2
    plot(f(:,1),f(:,2),'g+')
end

if n_obj == 3
    scatter3(f(:,1),f(:,2),f(:,3),'g+')
end

feval([domain '_moref'],1);