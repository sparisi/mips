function [dataset, J] = collect_samples_ctx(domain, ...
    maxepisodes, maxsteps, policy, context)
% COLLECT_SAMPLES_CTX Collects full contextual episodes, i.e., it returns a 
% DATASET containing information also about the single steps for each 
% episode.

simulator = [domain '_simulator'];
mdpvars = feval([domain '_mdpvariables']);
maxr = mdpvars.maxr;

empty_sample.s = [];
empty_sample.a = [];
empty_sample.r = [];
empty_sample.nexts = [];
empty_sample.terminal = [];
dataset = repmat(empty_sample, 1, maxepisodes);

J = 0;

% Main loop
parfor episodes = 1 : maxepisodes
    
    dataset(episodes).policy = policy;
    
    % Select initial state
    initial_state = feval(simulator);
    
    % Run one episode (up to the max number of steps)
    [sample, totrew] = ...
        execute_ctx(domain, initial_state, simulator, policy, maxsteps, context);
    J = J + totrew;
    
    % Normalize rewards
    sample.r = bsxfun(@times,sample.r,1./maxr);
    
    % Store the new sample
    if ~isempty(sample.a)
        dataset(episodes).s = sample.s;
        dataset(episodes).a = sample.a;
        dataset(episodes).r = sample.r;
        dataset(episodes).nexts = sample.nexts;
        dataset(episodes).terminal = sample.terminal;
    end
    
end

J = J / maxepisodes;

return
