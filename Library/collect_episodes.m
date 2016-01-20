function [J, Theta, Context] = collect_episodes(mdp, episodes, steps, pol_high, pol_low)
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
%     - J        : returns averaged over all the episodes
%     - Theta    : parameters of the low level policy sampled from the high
%                  level one
%     - Context  : (only for contextual MDPs)

pol_low = repmat(pol_low,1,episodes);

if ismember('CMDP',superclasses(mdp)) % Contextual MDP
    Context = mdp.getcontext(episodes);
    Theta = pol_high.drawAction(Context);
    for i = 1 : episodes
        pol_low(i) = pol_low(i).update(Theta(:,i));
    end
    J = evaluate_policies(mdp, 1, steps, pol_low, Context);
    
else
    Theta = pol_high.drawAction(episodes);
    for i = 1 : episodes
        pol_low(i) = pol_low(i).update(Theta(:,i));
    end
    J = evaluate_policies(mdp, 1, steps, pol_low);
    
end
