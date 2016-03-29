% Script for comparing the results of different methods averaged over 
% several trials. Data is in files named 'METHOD_TRIAL.mat' (1_1.mat,
% 1_2.mat, ..., 2_2.mat, etc...).

close all
clear all

for METHOD = [1 3 4 10 11]
    
    counter = 1;
    for TRIAL = 1 : 999
        try
            load([num2str(METHOD) '_' num2str(TRIAL)])
            dataMatrix(counter,:) = J_history; % Change to your desired data
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
        {'LineWidth', 2, 'DisplayName', num2str(METHOD)}, ...
        0.1 );
    legend('-DynamicLegend');
    
end