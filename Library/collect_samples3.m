function [data, J] = collect_samples3(mdp, minsamples, maxsteps, policy)
% COLLECT_SAMPLES3 This version collects samples until at least MINSAMPLES 
% are collected, but looping over COLLECT_SAMPLES.
%
%    INPUT
%     - mdp        : the MDP to be solved
%     - minsamples : number of minimum samples to collect
%     - maxsteps   : max number of steps per episode
%     - policy     : policy followed by the agent
%
%    OUTPUT
%     - data       : see COLLECT_SAMPLES
%     - J          : returns averaged over all the episodes

episodes = floor(minsamples / maxsteps);
[data, J] = collect_samples(mdp, episodes, maxsteps, policy);
totsamples = size([data.r],2);
totepisodes = numel(data);
J = J*totepisodes;

while totsamples < minsamples
    
    episodes = ceil((minsamples - totsamples) / maxsteps);
    [data_i, J_i] = collect_samples(mdp, episodes, maxsteps, policy);
    totsamples = totsamples + size([data_i.r],2);
    data = vertcat(data,data_i);
    episodes_i = numel(data_i);
    J = J + J_i*episodes_i;
    totepisodes = totepisodes + episodes_i;

end

J = J / totepisodes;
