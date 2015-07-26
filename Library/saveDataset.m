function saveDataset( filename, data, type, reward_idx, continuous )
% Writes a dataset on a file.
%
% Inputs:
%
%  - filename   : name of the output file
%  - data       : dataset to be saved
%  - type       : 1 <state, action, reward, endep> (DEFAULT)
%                 2 <state, action, nextstate, reward>
%  - reward_idx : indices of the rewards to be saved (DEFAULT all)
%  - continuous : 0 episodic (DEFAULT). Next state of the last transition 
%                   is replaced by 123.456
%                 1 non episodic (continuous)

if nargin < 3
    type = 1;
    reward_idx = 1 : length(data(1).r(:,1));
    continuous = 0;
elseif nargin < 4
    reward_idx = 1 : length(data(1).r(:,1));
    continuous = 0;
elseif nargin < 5
    continuous = 0;
end
delete(filename);
nepisodes = size(data,2);

if type == 2
	dlmwrite(filename, ...
        [length(data(1).s(:,1)) + length(data(1).a(:,1)) length(data(1).s(:,1))+length(reward_idx)], ...
        '-append', 'delimiter', '\t', 'precision', 8);
end

for ep = 1 : nepisodes
    nsamples = size(data(ep).s, 2);
    for sample = 1 : nsamples
        state = data(ep).s(:,sample);
        statedim = length(data(ep).s(:,sample));
        action = data(ep).a(:,sample);
        reward = data(ep).r(:,sample);
        endep = data(ep).terminal(:,sample);
        
        if type == 1
            dlmwrite(filename, [state', action', reward(reward_idx)', endep], ...
                '-append', 'delimiter', '\t', 'precision', 8);
        else
            if endep == 1 && continuous == 0
                nextstate = 123.456 * ones(statedim,1);                
            else
                nextstate = data(ep).nexts(:,sample);
            end
            dlmwrite(filename, [state', action', nextstate', reward(reward_idx)'], ...
                '-append', 'delimiter', '\t', 'precision', 8);
        end
    end
end

return
