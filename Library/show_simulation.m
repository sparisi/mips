function [J, ds] = show_simulation(mdp, policy, pausetime, makedet)
% SHOW_SIMULATION Runs an episode and shows what happened during its 
% execution with an animation.
%
%    INPUT
%     - mdp       : the MDP to be seen
%     - policy    : the low level policy
%     - pausetime : time between plotting frames
%     - makedet   : 1 to make the policy deterministic, 0 otherwise
%
%    OUTPUT
%     - J         : the total return of the episode
%     - ds        : the episode dataset

if makedet, policy = policy.makeDeterministic; end

mdp.closeplot
[ds, J] = collect_samples(mdp, 1, 10000, policy);
mdp.plotepisode(ds, pausetime)