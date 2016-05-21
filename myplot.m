% Script for comparing results of different methods averaged over many trials. 
% Data is stored in files named 'METHOD_TRIAL.mat' (1_1.mat, 1_2.mat, ..., 2_2.mat, etc...).

close all
clear all

%% Change this values according to your needs
dataFOLDER = 'results/';
methodsALL = [1 3 4];
methodsNAMES = {'Method1', 'OtherMethod', 'SuperAlg'};
entryNAME = 'J_history';

%% Plot
for method = methodsALL;
    
    counter = 1;
    dataMatrix = [];
    for trial = 1 : 999
        try
            load([dataFOLDER num2str(method) '_' num2str(trial)])
            dataMatrix(counter,:) = eval(entryNAME);
            counter = counter + 1;
        catch
        end
    end
    % dataMatrix = moving(dataMatrix',2)';
    
    hold all
    shadedErrorBar( ...
        1:size(dataMatrix,2), ...
        mean(dataMatrix,1), ...
        std(dataMatrix), ...
        {'LineWidth', 2, 'DisplayName', methodsNAMES{methodsALL==method}}, ...
        0.1 );
    legend('-DynamicLegend');
    
end