function [J, ds] = see_policy(mdp, policy, policy_high, pausetime)
% SEE_POLICY Runs an episode and shows what happened during its execution
% with an animation. Any kind of stochasticiy is removed from the policy,
% but not from the environment.
%
%    INPUT
%     - mdp         : the MDP to be seen
%     - policy      : the low level policy
%     - policy_high : (optional) the high level policy that draws the low
%                     level policy parameters
%     - pausetime   : time between plotting frames
%
%    OUTPUT
%     - J           : the total return of the episode
%     - ds          : the episode dataset

if nargin < 4, pausetime = 0.5; end
if nargin > 2, policy = policy.makeDeterministic.update( ...
        policy_high.makeDeterministic.drawAction(1)); end

mdp.closeplot

[ds, J] = collect_samples(mdp, 1, 1000, policy);

mdp.plotepisode(ds,pausetime)