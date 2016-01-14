function [dataset, J, G, H] = collect_samples_rele(mdp, episodes, maxsteps, policy, context)
% COLLECT_SAMPLES_RELE Collects samples and computes gradients and hessians
% using ReLe.
% More info at https://github.com/AIRLab-POLIMI/ReLe
% See also the README for details.

gamma = mdp.gamma;
mexParams.nbRewards = mdp.dreward;
if nargin == 5, mexParams.ctx = context; end

switch class(mdp)
    
    case 'Dam'
        domain = 'dam';
        mexParams.penalize = mdp.penalize;
        if mexParams.penalize == 0
            mexParams.initType = 'random_discrete';
        else
            mexParams.initType = 'random';
        end
        
    case 'Lqr'
        domain = 'lqr';
        mexParams.dim = mdp_var.dim;
        mexParams.stddev = 1;
        
    case 'DeepSeaTreasure'
        domain = 'deep';
        
    otherwise
        error('Domain not implemented in ReLe.')
        
end

mexParams.policyParameters = policy.theta;

if nargout == 4
    [dataset, J, G, H] = collectSamples(domain, episodes, maxsteps, gamma, mexParams);

elseif nargout == 3
    [dataset, J, G] = collectSamples(domain, episodes, maxsteps, gamma, mexParams);

else
    [dataset, J] = collectSamples(domain, episodes, maxsteps, gamma, mexParams);
end

J = J';
