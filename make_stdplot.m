% Compares results of different methods averaged over many trials. 
% You need to set the folder where the data is stored and the filename
% format (eg, alg1_1.mat, alg1_2.mat, ..., alg2_1.mat, ...)
%
% Plot mean with 95% confidence interval.

close all
clear all
figure()
h = {};

%% Change entries according to your needs
folder = './data/';
separator = '_';
filenames = {'alg1', 'alg2'};

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

legendnames = {};
if isempty(legendnames), legendnames = filenames; end
colors = {};
markers = {};

variable = 'J_history';
% variable = 'mean(J_history)';
% variable = '-log(-mean(J_history))';
title(variable, 'Interpreter', 'none')

%% Plot
name_idx = 1;
name_valid = [];
for name = filenames
    counter = 1;
    dataMatrix = [];
    for trial = 0 : 999
        try
            load([folder name{:} separator num2str(trial) '.mat'], variable)
            dataMatrix(counter,:) = eval(variable);
            counter = counter + 1;
        catch
        end
    end
    
    if ~isempty(dataMatrix)
%         dataMatrix = moving(dataMatrix',10)';
        hold all
        lineprops = { 'LineWidth', 3, 'DisplayName', name{:} };
        if ~isempty(colors)
            lineprops = {lineprops{:}, 'Color', colors{name_idx} };
        end
        if ~isempty(markers)
            lineprops = {lineprops{:}, 'Marker', markers{name_idx} };
        end
        tmp = shadedErrorBar( ...
            1:size(dataMatrix,2), ...
            mean(dataMatrix,1), ...
            1.96*std(dataMatrix)/sqrt(size(dataMatrix,1)), ...
            lineprops, ...
            0.1, 0 );
        h{end+1} = tmp.mainLine;
        name_valid(end+1) = name_idx;
    end
    name_idx = name_idx + 1;
    
end

legend([h{:}], legendnames{name_valid}, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];
