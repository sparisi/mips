function [nextstate, reward, absorb] = resource_simulator(state, action)

mdp_vars = resource_mdpvariables();

if nargin == 0
    
    nextstate = [5; 3; 0; 0];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

i = state(1);
j = state(2);

% Transition function
switch action
    case 1
        nextj = max(1,j-1);
        nextstate = [i; nextj];
    case 2
        nextj = min(mdp_vars.state_dim(2),j+1);
        nextstate = [i; nextj];
    case 3
        nexti = max(1,i-1);
        nextstate = [nexti; j];
    case 4
        nexti = min(mdp_vars.state_dim(1),i+1);
        nextstate = [nexti; j];
    otherwise
        error('Unknown action.');
end

% Update position
nextstate(3) = state(3);
nextstate(4) = state(4);
% Update gems and gold
if nextstate(1) == 1 && nextstate(2) == 3
    nextstate(3) = 1;
elseif nextstate(1) == 2 && nextstate(2) == 5
    nextstate(4) = 1;
end

% Reward functions
reward1 = 0;
fight = 0;
if (nextstate(1) == 1 && nextstate(2) == 4) || ...
        (nextstate(1) == 2 && nextstate(2) == 3)
    r = rand();
    if r < 0.1
        fight = 1;
    else
        fight = 0;
    end
end

if fight
    reward1 = -1; % Fight penalty
    state(3) = 0;
    state(4) = 0;
    nextstate(1) = 5;
    nextstate(2) = 3;
end

absorb = 0;

reward2 = 0;
reward3 = 0;
if(nextstate(1) == 5 && nextstate(2) == 3)
    reward2 = state(3); % Gold
    reward3 = state(4); % Gems
    nextstate(3) = 0;
    nextstate(4) = 0;
end

reward = [reward1; reward2; reward3] ./ mdp_vars.max_obj;

return
