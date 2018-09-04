% Compares results of different methods averaged over many trials. 
% You need to set the folder where the data is stored and the filename
% format (eg, alg1_1.mat, alg1_2.mat, ..., alg2_1.mat, ...)

close all
clear all

%% Change entries according to your needs
folder = './data_ppo_slow/';
separator = '_';
filenames = {'ddpg.ddpg0_Pendulum-v0', 'ddpg.ddpg_tdreg1_Pendulum-v0', 'ddpg.ddpg_tdreg2_Pendulum-v0', 'ddpg.ddpg_tdreg3_Pendulum-v0', 'ddpg.ddpg_tdreg4_Pendulum-v0'};
filenames = {'ddpg.ddpg_tdreg1_Pendulum-v0', 'ddpg.ddpg_tdreg3_Pendulum-v0', 'ddpg.ddpg_tdreg4_Pendulum-v0'};
filenames = {'ddpg.ddpg0_Pendulum-v0', 'ddpg.ddpg_tdreg4_Pendulum-v0'};
filenames = {'ppo.ppo0_HalfCheetah-v2', 'ppo.ppo1_HalfCheetah-v2'};
filenames = {};

if isempty(filenames) % automatically identify algorithms name
    allfiles = dir(fullfile(folder,'*.mat'));
    for i = 1 : length(allfiles)
        tmpname = allfiles(i).name(1:end-4); % remove .mat from string
        trial_idx = strfind(tmpname, separator); % find separator
        tmpname = tmpname(1:trial_idx(end)-1); % remove trial idx from string
        if isempty(filenames) || ~strcmp(filenames{end}, tmpname) % if new name, add it
            filenames{end+1} = tmpname;
        end
    end
end

legendnames = {}; % if empty, filenames will be used
colors = {};
markers = {};

variable = 'J_history';
% variable = 'mean(J_history)';
% variable = '-log(-mean(J_history))';

%% Plot
h = {};
for name = filenames
    
    counter = 1;
    dataMatrix = [];
    for trial = 0 : 999
        try
            load([folder name{:} separator num2str(trial) '.mat'])
            dataMatrix(counter,:) = eval(variable);
            counter = counter + 1;
        catch
        end
    end
    
    if ~isempty(dataMatrix)
%     dataMatrix = moving(dataMatrix',10)';
        hold all
        lineprops = { 'LineWidth', 2, 'DisplayName', name{:} };
        if ~isempty(colors)
            lineprops = {lineprops{:}, 'Color', colors{numel(h)+1} };
        end
        if ~isempty(markers)
            lineprops = {lineprops{:}, 'Marker', markers{numel(h)+1} };
        end
        tmp = shadedErrorBar( ...
            1:size(dataMatrix,2), ...
            mean(dataMatrix,1), ...
            std(dataMatrix), ...
            lineprops, ...
            0.1 );
        h{end+1} = tmp.mainLine;
    end
    
end

legend([h{:}], legendnames, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];
