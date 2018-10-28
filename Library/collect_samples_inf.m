function [data, J] = collect_samples_inf(mdp, minsamples, reset_prob, policy)
% COLLECT_SAMPLES_INF Runs single "neverending" episodes until at least 
% MINSAMPLES are collected. An episode resets randomly with probability
% PROB_RESET, or if a terminal state occurs. If MINSAMPLES have been 
% collected when a reset occurs, the data collection is over.
%
%    INPUT
%     - mdp        : the MDP to be solved
%     - minsamples : number of minimum samples to collect
%     - reset_prob : probability of resetting the MDP
%     - policy     : policy followed by the agent
%
%    OUTPUT
%     - data       : struct with the following fields (one per episode)
%                     * s        : state
%                     * a        : action
%                     * nexts    : next state
%                     * r        : immediate reward
%                     * endsim   : 1 if the state is terminal or a reset 
%                                  happens, 0 otherwise
%                     * length   : length of the episode
%                     * episodes : number of episodes collected
%     - J          : returns averaged over all the episodes

data.s = nan(mdp.dstate, 0);
data.nexts = nan(mdp.dstate, 0);
data.a = nan(mdp.daction, 0);
data.r = nan(mdp.dreward, 0);
data.endsim = nan(1, 0);
data.length = nan(0, 0);

totsamples = 0;
step = 0;
J = 0;
episodes = 1;
state = mdp.initstate(1);

while true
    
    totsamples = totsamples + 1;
    step = step + 1;
    action = policy.drawAction(state);
    [nextstate, reward, endsim] = mdp.simulator(state, action);
    endsim = endsim || rand() < reset_prob;
    
    data.a(:,end+1) = action;
    data.r(:,end+1) = reward;
    data.s(:,end+1) = state;
    data.nexts(:,end+1) = nextstate;
    data.endsim(:,end+1) = endsim;

    J(episodes) = J(episodes) + mdp.gamma^(step-1)*reward;
    
    if endsim
        data.length(:,end+1) = step;
        if totsamples >= minsamples
            break
        else
            state = mdp.initstate(1);
            episodes = episodes + 1;
            J(episodes) = 0;
            step = 0;
        end
    end

end

data.episodes = episodes;

J = mean(J);
