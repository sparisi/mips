function J = evaluate_policies_rele(mdp, episodes, maxsteps, policies, context)
% EVALUATE_POLICIES_RELE Evaluates a set of policies using ReLe.
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

%%
J = zeros(mdp.dreward,numel(policies));
for i = 1 : numel(policies)
    mexParams.policyParameters = policies(i).theta;
    [~, J] = collectSamples(domain, episodes, maxsteps, gamma, mexParams);
    J(:,i) = mean(J,1);
end