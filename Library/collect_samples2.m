function data = collect_samples2(mdp, episodes, maxsteps, policy)
% COLLECT_SAMPLES2 Faster version of COLLECT_SAMPLES. It is not possible to
% distinguish between episodes.
%
%    INPUT
%     - mdp      : the MDP to be solved
%     - episodes : number of episodes to run
%     - maxsteps : max number of steps per episode
%     - policy   : policy.drawAction is a function receiving states and
%                  returning actions
%
%    OUTPUT
%     - data     : struct with the following fields
%                   * s        : states
%                   * a        : actions
%                   * nexts    : next states
%                   * r        : immediate rewards
%                   * terminal : 1 if the state is terminal, 0 otherwise

data.s = nan(mdp.dstate, 0);
data.nexts = nan(mdp.dstate, 0);
data.a = nan(mdp.daction, 0);
data.r = nan(mdp.dreward, 0);
data.terminal = nan(1, 0);

ongoing = true(1,episodes);
step = 0;
state = mdp.initstate(episodes);

while ( (step < maxsteps) && sum(ongoing) > 0 )
    
    step = step + 1;
    running_states = state(:,ongoing);
    n = sum(ongoing);
    action = policy.drawAction(running_states);
    
    [nextstate, reward, terminal] = mdp.simulator(running_states, action);
    
    data.a(:,end+1:end+n) = action;
    data.r(:,end+1:end+n) = reward;
    data.s(:,end+1:end+n) = running_states;
    data.nexts(:,end+1:end+n) = nextstate;
    data.terminal(:,end+1:end+n) = terminal;
    
    state(:,ongoing) = nextstate;
    ongoing(ongoing) = ~terminal;
    
end
