function [data, avgRew] = collect_episodes(mdp, episodes, steps, pol_high, pol_low)
% COLLECT_EPISODES Collects episodes, meant as pairs (J,Theta). Each 
% parameter vector Theta(:,i) identifies a low level policy and is drawn 
% from the high level policy. The parameters are kept fixed during the 
% whole episode.
%
% See also EVALUATE_POLICIES.
%
%    INPUT
%     - mdp      : the MDP to be solved
%     - episodes : episodes to collect
%     - maxsteps : max number of steps per episode
%     - pol_low  : low level policy
%     - high_pol : high level policy
%
%    OUTPUT
%     - data     : struct with the following fields
%                   * J       : [R x N] matrix with the return of each
%                               episode (R is the number of objective, N 
%                               the number of episodes)
%                   * Theta   : [D x N] matrix with the policy parameters
%                               drawn at each episode
%                   * Context : (optional) [C x N] matrix with the context
%                               of each episode
%     - avgRew   : return averaged over all the episodes

pol_low = repmat(pol_low,1,episodes);

if ismember('CMDP',superclasses(mdp)) % Contextual MDP
    data.Context = mdp.getcontext(episodes);
    data.Theta = pol_high.drawAction(data.Context);
    for i = 1 : episodes
        pol_low(i) = pol_low(i).update(data.Theta(:,i));
    end
    data.J = evaluate_policies(mdp, 1, steps, pol_low, data.Context);
    
else
    data.Theta = pol_high.drawAction(episodes);
    for i = 1 : episodes
        pol_low(i) = pol_low(i).update(data.Theta(:,i));
    end
    data.J = evaluate_policies(mdp, 1, steps, pol_low);
    
end

avgRew = mean(data.J,2);