function [nextstate, reward, absorb] = resource_simulator(state, action)

[hasGems, hasGold, hasEnemy] = resource_environment;
[nrows, ncols] = size(hasGems);

if nargin == 0
    
    nextstate = [5; 3; 0; 0];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

i = state(1);
j = state(2);

% Transition function (coordinates are (Y,X)!)
switch action
    case 1 % Left
        nextstate = [i; max(1,j-1); 0; 0];
    case 2 % Right
        nextstate = [i; min(ncols,j+1); 0; 0];
    case 3 % Up
        nextstate = [max(1,i-1); j; 0; 0];
    case 4 % Down
        nextstate = [min(nrows,i+1); j; 0; 0];
    otherwise
        error('Unknown action.');
end

% Update gems and gold
nextstate(3) = state(3) || hasGems(nextstate(1),nextstate(2));
nextstate(4) = state(4) || hasGold(nextstate(1),nextstate(2));

% Check fight
reward1 = 0;
if hasEnemy(nextstate(1),nextstate(2)) && rand() < 0.1
    reward1 = -1; % Fight penalty
    nextstate(3) = 0; % Lose gems
    nextstate(4) = 0; % Lose gold
    nextstate(1) = 5; % Back to init pos
    nextstate(2) = 3;
end

% Check for gold and gems rewards
reward2 = 0;
reward3 = 0;
if(nextstate(1) == 5 && nextstate(2) == 3)
    reward2 = nextstate(3); % Gold
    reward3 = nextstate(4); % Gems
    nextstate(3) = 0;
    nextstate(4) = 0;
end

reward = [reward1; reward2; reward3];
absorb = 0; % Infinite horizon

return
