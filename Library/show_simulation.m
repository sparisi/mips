function [J, ds] = show_simulation(mdp, policy, pausetime, steps)
% SHOW_SIMULATION Runs an episode and shows what happened during its 
% execution with an animation.
%
%    INPUT
%     - mdp       : the MDP to be seen
%     - policy    : the low level policy
%     - steps     : steps of the episode
%     - pausetime : time between animation frames
%
%    OUTPUT
%     - J         : the total return of the episode
%     - ds        : the episode dataset

mdp.closeplot
[ds, J] = collect_samples(mdp, 1, steps, policy);
mdp.plotepisode(ds, pausetime)