function J = evaluate_policies_high(mdp, episodes, maxsteps, policy_low, policies_high)
% EVALUATE_POLICIES_HIGH Evaluates a set of high level policies. For each 
% policy, several episodes are simulated.
%
%    INPUT
%     - mdp           : the MDP to be solved
%     - episodes      : number of episodes per policy
%     - maxsteps      : max number of steps per episode
%     - policy_low    : low level policy (its parameters are drawn from the
%                       high level policies)
%     - policies_high : high level policies to be evaluated
%
%    OUTPUT
%     - J             : average returns of each policy

npolicies = numel(policies_high);
J = zeros(mdp.dreward,npolicies);

parfor i = 1 : npolicies
    J_ep = collect_episodes(mdp, episodes, maxsteps, policies_high(i), policy_low);
    J(:,i) = mean(J_ep,2);
end
