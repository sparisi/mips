function [nextstate, reward, absorb] = deep_simulator(state, action)

if nargin == 0
    
    % Initial state
    nextstate = [1; 1];
    return
    
elseif nargin == 1
    
    nextstate = state;
    return
    
end

[treasure, isWhite] = deep_environment();
[nrows, ncols] = size(isWhite);
i = state(1);
j = state(2);

% Transition function (coordinates are (Y,X)!)
switch action
    case 1 % Left
        nextj = max(1,j-1);
        if ~isWhite(i,nextj)
            nextj = j;
        end
        nextstate = [i; nextj];
    case 2 % Right
        nextj = min(ncols,j+1);
        if ~isWhite(i,nextj)
            nextj = j;
        end
        nextstate = [i; nextj];
    case 3 % Up
        nexti = max(1,i-1);
        if ~isWhite(nexti,j)
            nexti = i;
        end
        nextstate = [nexti; j];
    case 4 % Down
        nexti = min(nrows,i+1);
        if ~isWhite(nexti,j)
            nexti = i;
        end
        nextstate = [nexti; j];
    otherwise
        error('Unknown action.');
end

reward1 = treasure(nextstate(1), nextstate(2)); % Treasure value
reward2 = -1; % Time
reward = [reward1; reward2]; 
absorb = (reward1 > 0);

return