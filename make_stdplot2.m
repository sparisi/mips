% Compares results of different hyperparameters (of the same method) 
% averaged over many trials. 
% You need to set the folder (where data is stored) and the filename
% format (eg, alg_param1_param2_1.mat)

close all
clear all
figure()
h = {};

%% Change entries according to your needs
folder = 'data/';
base_name = 'alg'; title(base_name, 'Interpreter', 'none')
separator = '_';

hp1_list = {0.1, 0.5, 1}; % hyperparameter lists
hp2_list = {0.01, 0.1, 0.2};
hp3_list = {'False'};
legendnames = {};

variable = 'J_history';
% variable = 'mean(J_history)';
% variable = '-log(-mean(J_history))';

%% Plot
for hp1 = hp1_list
    for hp2 = hp2_list
        for hp3 = hp3_list

            
name = [num2str(hp1{:}), separator, num2str(hp2{:}), separator, num2str(hp3{:})];
counter = 1;
dataMatrix = [];
for trial = 1 : 999
    try
        load([folder name separator num2str(trial) '.mat'], variable)
        dataMatrix(counter,:) = eval(variable);
        counter = counter + 1;
    catch
    end
end

if ~isempty(dataMatrix)
%     dataMatrix = moving(dataMatrix',2)';
    hold all
    tmp = shadedErrorBar( ...
        1:size(dataMatrix,2), ...
        mean(dataMatrix,1), ...
        1.96*std(dataMatrix)/sqrt(size(dataMatrix,1)), ...
        { 'LineWidth', 3, 'DisplayName', name }, ...
        0.1, 0 );
        h{end+1} = tmp.mainLine;
    legendnames = [legendnames, name];
end


        end
    end
end

legend([h{:}], legendnames, 'Interpreter', 'none')

leg.Position = [0.2 0.7 0 0];
