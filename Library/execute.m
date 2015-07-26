function [new_results, totrew, totentropy] = execute( ...
    domain, initial_state, simulator, policy, maxsteps )
  
  % Get MDP characteristics
  mdpconfig = [domain '_mdpvariables'];
  mdp_vars = feval(mdpconfig);
  nvar_state = mdp_vars.nvar_state;
  nvar_action = mdp_vars.nvar_action;
  nvar_reward = mdp_vars.nvar_reward;
  gamma = mdp_vars.gamma;
  is_avg = mdp_vars.is_avg;
  
  % Initialize variables
  totrew = zeros(nvar_reward,1);
  totentropy = 0;
  steps = 0;
  endsim = 0;
  
  % Set initial state
  state = initial_state;
  
  % Allocate memory for new samples 
  results.s = -9999*ones(nvar_state, maxsteps+1);
  results.nexts = -9999*ones(nvar_state, maxsteps+1);
  results.a = -9999*ones(nvar_action, maxsteps+1);
  results.r = -9999*ones(nvar_reward, maxsteps+1);
  results.terminal = -9999*ones(1, maxsteps+1);
  results.s(:,1) = state;

  % Run the episodes
  while ( (steps < maxsteps) && (~endsim) )

    steps = steps + 1;

    % Select action 
    action = policy.drawAction(state);
    
    % Compute entropy
    totentropy = totentropy + policy.entropy(state);

    % Simulate
    [nextstate, reward, endsim] = feval(simulator, state, action);
    
    % Record sample
 	results.a(:,steps) = action;
 	results.r(:,steps) = reward;
	results.s(:,steps+1) = nextstate;
 	results.nexts(:,steps) = nextstate;
    results.terminal(:,steps) = endsim;
    
    % Update the total reward
    totrew = totrew + (gamma)^(steps-1) * reward;

    % Continue
    state = nextstate;
    
  end
    
  % Return the results
  new_results.s = results.s(:, 1:steps);
  new_results.a = results.a(:, 1:steps);
  new_results.r = results.r(:, 1:steps);
  new_results.nexts = results.nexts(:, 1:steps);
  new_results.terminal = results.terminal(:, 1:steps);
  
  % If we are in the average reward setting, then normalize the return
  if is_avg && gamma == 1
      totrew = totrew / steps;
  end
  
  totentropy = totentropy / steps;
  
  return
  
