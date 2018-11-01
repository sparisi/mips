function [data, J] = collect_samples_inf(mdp, minsamples, reset_prob, policy)
% COLLECT_SAMPLES_INF Runs single "neverending" episodes until at least 
% MINSAMPLES are collected. An episode resets randomly with probability
% RESET_PROB, or if a terminal state occurs. If MINSAMPLES have been 
% collected when a reset occurs, the data collection is over.
%
%    INPUT
%     - mdp        : the MDP to be solved
%     - minsamples : number of minimum samples to collect
%     - reset_prob : probability of resetting the MDP
%     - policy     : policy followed by the agent
%
%    OUTPUT
%     - data       : struct with the following fields
%                     * s        : state
%                     * a        : action
%                     * nexts    : next state
%                     * r        : immediate reward
%                     * terminal : 1 if the state is terminal or a reset 
%                                  happens, 0 otherwise
%                     * length   : length of each episode
%                     * t        : time index
%                     * episodes : number of episodes collected
%     - J          : expected returns averaged over all episodes

data.s = nan(mdp.dstate, 0);
data.nexts = nan(mdp.dstate, 0);
data.a = nan(mdp.daction, 0);
data.r = nan(mdp.dreward, 0);
data.terminal = nan(1, 0);
data.t = nan(1, 0);
data.length = nan(0, 0);

totsamples = 0;
step = 0;
J = 0;
episode = 1;
state = mdp.initstate(1);

while true
    
    totsamples = totsamples + 1;
    step = step + 1;
    action = policy.drawAction(state);
    [nextstate, reward, terminal] = mdp.simulator(state, action);
    terminal = terminal || rand() < reset_prob;
    
    data.a(:,end+1) = action;
    data.r(:,end+1) = reward;
    data.s(:,end+1) = state;
    data.nexts(:,end+1) = nextstate;
    data.terminal(:,end+1) = terminal;
    data.t(:,end+1) = step;
    
    state = nextstate;

    J(episode) = J(episode) + mdp.gamma^(step-1)*reward;
    
    if terminal
        data.length(:,end+1) = step;
        if mdp.isAveraged && mdp.gamma == 1
            J(episode) = J(episode) / step;
        end
        if totsamples >= minsamples
            break
        else
            state = mdp.initstate(1);
            episode = episode + 1;
            J(episode) = 0;
            step = 0;
        end
    end

end

data.episodes = episode;

J = mean(J);
