function [dataset, J, G, H] = collect_samples_rele(domain, ...
    maxepisodes, maxsteps, policy)
% COLLECT_SAMPLES_RELE Collects samples and computes gradients and hessians
% using ReLe.
% More info at https://github.com/AIRLab-POLIMI/ReLe
% See also the README for details.

mdp_var = feval([domain '_mdpvariables']);
gamma = mdp_var.gamma;
n_obj = mdp_var.nvar_reward;

mexParams.nbRewards = n_obj;

switch domain
    
    case 'dam'
        mexParams.penalize = mdp_var.penalize;
        if mexParams.penalize == 0
            mexParams.initType = 'random_discrete';
        else
            mexParams.initType = 'random';
        end
        
    case 'lqr'
        mexParams.dim = mdp_var.dim;
        mexParams.stddev = 1;
        
    case 'deep'
        
    otherwise
        error('Domain not implemented.')
        
end

mexParams.policyParameters = policy.theta;

if nargout == 4
    [dataset, J, G, H] = collectSamples(domain, maxepisodes, maxsteps, gamma, mexParams);

elseif nargout == 3
    [dataset, J, G] = collectSamples(domain, maxepisodes, maxsteps, gamma, mexParams);

else
    [dataset, J] = collectSamples(domain, maxepisodes, maxsteps, gamma, mexParams);
end

J = mean(J,1);
