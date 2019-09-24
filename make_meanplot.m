% Compares results of different methods averaged over many trials. 
% You need to set the folder where the data is stored and the filename
% format (eg, alg1_1.mat, alg1_2.mat, ..., alg2_1.mat, ...)
%
% Plot only mean.

close all
clear all
figure()
h = {};
plot_every = 5; % plots data with the following indices [1, 5, 10, ..., end]
mov_avg = 20; % computes moving avg to smooth the plot

%% Change entries according to your needs
folder = '.data';
separator = '_';
filenames = {'alg1', 'alg2'};

if isempty(filenames) % automatically identify algorithms name
    allfiles = dir(fullfile(folder,'*.mat'));
    for i = 1 : length(allfiles)
        tmpname = allfiles(i).name(1:end-4); % remove .mat from string
        trial_idx = strfind(tmpname, separator); % find separator
        tmpname = tmpname(1:trial_idx(end)-1); % remove trial idx from string
        if (isempty(filenames) || ~strcmp(filenames{end}, tmpname) ) && ~any(strcmp(filenames, tmpname)) % if new name, add it
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
            load([folder '/' name{:} separator num2str(trial) '.mat'], variable)
            tmp = eval(variable);
            if counter == 1
                dataMatrix(counter,:) = tmp;
            else
                len = min(size(dataMatrix,2), length(tmp));
                dataMatrix = dataMatrix(:,1:len);
                dataMatrix(counter,:) = tmp(1:len);
            end
            counter = counter + 1;
        catch
        end
    end
    
    if ~isempty(dataMatrix)
        dataMatrix = moving(dataMatrix',mov_avg)';
        hold all
        lineprops = { 'LineWidth', 3, 'DisplayName', name{:} };
        if ~isempty(colors)
            lineprops = {lineprops{:}, 'Color', colors{name_idx} };
        end
        if ~isempty(markers)
            lineprops = {lineprops{:}, 'Marker', markers{name_idx} };
        end
        m = mean(dataMatrix,1);
        x = [1, plot_every : plot_every : length(m)];
        h{end+1} = plot(x, m(x), lineprops{:});
        name_valid(end+1) = name_idx;
    else
        disp([name{:} ' is empty!'])
    end
    name_idx = name_idx + 1;
    
end

legend([h{:}], {legendnames{name_valid}}, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];
