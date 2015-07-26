%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finds the Pareto-frontier using the Radial Algorithm and the Natural 
% Gradient.
% 
% Reference: S Parisi, M Pirotta, N Smacchia, L Bascetta, M Restelli (2014)
% Policy gradient approaches for multi-objective sequential decision making
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

verboseOut = false; % to print output messages

domain = 'dam';
makeDet = 0; % if we want to consider stochastic or deterministic policies
[N_obj, init_pol, episodes, steps, gamma] = feval([domain '_settings']);

N_params = length(init_pol.theta);

simstep = 30; % density of the directions in the simplex
tolerance = 0.001; % tolerance for the norm of the gradient
maxIter = 200; % max number of policy gradient steps in the same direction
minS = 1.5; % min entropy of the policy (with Gaussian policies the (differential) entropy can be negative)
lrate = 4;

% generate all combinations of weights for the directions in the simplex
W = convexWeights(N_obj, simstep);
N_sol = size(W,1);
front_pol = cell(N_sol,1);
inter_pol = [];

% initial solution
[ds, J_init] = collect_samples(domain, episodes, steps, init_pol);
M_init = zeros(N_params,N_obj);
for j = 1 : N_obj
    [nat_grad, stepsize] = eNACbase(init_pol,ds,gamma,j,lrate);
    M_init(:,j) = nat_grad * stepsize;
end

str_init = strtrim(sprintf('%.4f, ', J_init));
str_init(end) = [];
if verboseOut, fprintf('Starting from solution: [ %s ] \n', str_init); end

n_iterations = 0; % total number of iterations (policy evaluations)

%% For all the directions in the simplex
parfor k = 1 : N_sol
    
    fixedLambda = W(k,:)';
    newDir = M_init * fixedLambda;
    curr_pol = init_pol.update(lrate * newDir);

    iter = 1;
    str_dir = strtrim(sprintf('%.3f, ', fixedLambda));
    str_dir(end) = [];

    %% Policy gradient learning in one fixed direction
    while true
        
        n_iterations = n_iterations + 1;

        [ds, J, S] = collect_samples(domain, episodes, steps, curr_pol);

        M = zeros(N_params,N_obj); % jacobian
        Mn = zeros(N_params,N_obj); % jacobian with normalized gradients
        for j = 1 : N_obj
            [nat_grad, stepsize] = eNACbase(curr_pol,ds,gamma,j, lrate);
            M(:,j) = nat_grad * stepsize;
            Mn(:,j) = nat_grad / max(norm(nat_grad),1e-8);
        end

        dir = M * fixedLambda;
        dev = norm(dir);

        str_obj = strtrim(sprintf('%.4f, ', J));
        str_obj(end) = [];
        if verboseOut, fprintf('[ %s ] || Iter %d ) Norm: %.4f, S: %.2f, J: [ %s ] \n', ...
            str_dir, iter, dev, S, str_obj); end

        if dev < tolerance
            if verboseOut, fprintf('Cannot proceed any further in this direction!\n'); end
            front_pol{k} = curr_pol;
            break;
        end

        % for the min-norm Pareto-ascent direction, always use Mn
        dirPareto = paretoDirection(N_obj, Mn);
        devPareto = norm(dirPareto);
        
        if devPareto < tolerance
            if verboseOut, fprintf('Pareto front reached!\n'); end
            front_pol{k} = curr_pol;
            break;
        end
        
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
        
        inter_pol = [inter_pol; curr_pol];
        iter = iter + 1;

        curr_pol = curr_pol.update(dir);

    end
    
end

front_pol = vertcat(front_pol{:});

%% Plot
f = evaluate_policies(front_pol, domain, makeDet);
[f, p] = pareto(f, front_pol);

figure; hold all
if N_obj == 2
    plot(f(:,1),f(:,2),'g+')
end

if N_obj == 3
    scatter3(f(:,1),f(:,2),f(:,3),'g+')
end

feval([domain '_moref'],1);
