function [data, J] = collect_samples(mdp, episodes, maxsteps, policy, contexts)
% COLLECT_SAMPLES Simulates episodes and provides low level details, i.e., 
% it collects tuples (s,a,s',r) at each time step.
% If you want to collect just the return of many different policies, please 
% see COLLECT_EPISODES.
%
%    INPUT
%     - mdp      : the MDP to be solved
%     - episodes : number of episodes to run
%     - maxsteps : max number of steps per episode
%     - policy   : policy followed by the agent
%     - contexts : (optional) contexts of each episode
%
%    OUTPUT
%     - data     : struct with the following fields (one per episode)
%                   * s       : state
%                   * a       : action
%                   * nexts   : next state
%                   * r       : immediate reward
%                   * gammar  : discounted immediate reward, gamma^(t-1)*r
%                   * Q       : approximation of the Q function, that is
%                               Q(s_t,a_t) = sum_(h=t)^T gamma^(h-1)*r(s_t,a_t)
%                   * length  : length of the episode
%     - J        : returns averaged over all the episodes

assert(numel(policy) == 1, ...
    ['This function supports only one policy as input.' ...
    'For collecting samples with multiple policies, see COLLECT_EPISODES.'])

% Get MDP characteristics
nvar_state = mdp.dstate;
nvar_action = mdp.daction;
nvar_reward = mdp.dreward;
gamma = mdp.gamma;
isAveraged = mdp.isAveraged;

% Initialize variables
totrew = zeros(nvar_reward,episodes);
step = 0;

% Initialize simulation
simulator = @mdp.simulator;
initial_state = mdp.initstate(episodes);
state = initial_state;

% Allocate memory
ds.s = nan(nvar_state, maxsteps, episodes);
ds.nexts = nan(nvar_state, maxsteps, episodes);
ds.a = nan(nvar_action, maxsteps, episodes);
ds.r = nan(nvar_reward, maxsteps, episodes);
ds.gammar = nan(nvar_reward, maxsteps, episodes);

% Keep track of the states which did not terminate
ongoing = true(1,episodes);

% Save the last step per episode
endingstep = maxsteps*ones(1,episodes);

% Run the episodes until maxsteps or all ends
while ( (step < maxsteps) && sum(ongoing) > 0 )
    
    step = step + 1;
    running_states = state(:,ongoing);
        
    % Select action
    action = policy.drawAction(running_states);
    
    % Simulate one step of all running episodes at the same time
    if nargin < 5
        [nextstate, reward, endsim] = feval(simulator, running_states, action);
    else
        [nextstate, reward, endsim] = feval(simulator, running_states, action, contexts(:,ongoing));
    end
    
    % Update the total reward
    totrew(:,ongoing) = totrew(:,ongoing) + (gamma)^(step-1) .* reward;
    
    % Record sample
    ds.a(:,step,ongoing) = action;
    ds.r(:,step,ongoing) = reward;
    ds.gammar(:,step,ongoing) = (gamma)^(step-1) .* reward;
    ds.s(:,step,ongoing) = running_states;
    ds.nexts(:,step,ongoing) = nextstate;
    
    % Continue
    idx = 1:episodes;
    idx = idx(ongoing);
    idx = idx(endsim);
    state(:,ongoing) = nextstate;
    ongoing(ongoing) = ~endsim;
    endingstep(idx) = step;
    
end

% Convert dataset to struct to allow storage of episodes with different length
data = struct( ...
    's', squeeze(num2cell(ds.s(:,1:step,:),[1 2])), ...
    'a', squeeze(num2cell(ds.a(:,1:step,:),[1 2])), ...
    'r', squeeze(num2cell(ds.r(:,1:step,:),[1 2])), ...
    'nexts', squeeze(num2cell(ds.nexts(:,1:step,:),[1 2])), ...
    'gammar', squeeze(num2cell(ds.gammar(:,1:step,:),[1 2])), ...
    'length', 1 ...
    );

for i = 1 : episodes
    data(i).s = data(i).s(:,1:endingstep(i));
    data(i).r = data(i).r(:,1:endingstep(i));
    data(i).gammar = data(i).r(:,1:endingstep(i));
    data(i).a = data(i).a(:,1:endingstep(i));
    data(i).nexts = data(i).nexts(:,1:endingstep(i));
    data(i).Q = cumsum(bsxfun(@times,data(i).r,gamma.^(0:data(i).length-1)),2,'reverse');
    data(i).length = endingstep(i);
end

% If we are in the average reward setting, then normalize the return
if isAveraged && gamma == 1, totrew = totrew / step; end

J = mean(totrew,2);

return
