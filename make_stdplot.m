% Script for comparing results of different methods averaged over many trials. 
% Data is stored in files named 'METHOD_TRIAL.mat' (1_1.mat, 1_2.mat, ..., 2_2.mat, etc...).

close all
clear all

%% Change entries according to your needs
folder = 'results/';
filenames = {'alg1', 'alg2', 'alg3'};
variable = 'J_history';
variable = 'mean(J_history)';
variable = '-log(-mean(J_history))';

%% Plot
for name = filenames
    
    counter = 1;
    dataMatrix = [];
    for trial = 1 : 999
        try
            load([folder name{:} '_' num2str(trial)])
            dataMatrix(counter,:) = eval(variable);
            counter = counter + 1;
        catch
        end
    end
    % dataMatrix = moving(dataMatrix',2)';
    
    if ~isempty(dataMatrix)
        hold all
        shadedErrorBar( ...
            1:size(dataMatrix,2), ...
            mean(dataMatrix,1), ...
            std(dataMatrix), ...
            { 'LineWidth', 2, 'DisplayName', name{:} }, ...
            0.1 );
        h(end+1) = tmp.mainLine;
    end
    
end

legend(h,name)

leg.Position = [0.2 0.7 0 0];