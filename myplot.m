% Script for plotting mean and std of some data from several trials. 
% In this example, trials are saved in files 'NAME1', 'NAME2', 'NAME3' ...

% close all
clear all

filename = 'NAME';
counter = 1;
for i = 1 : 999
    try
        load([filename num2str(i)])
        dataMatrix(counter,:) = J_history; % change to your desired data
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
    {'LineWidth', 2, 'DisplayName', filename}, ...
    0.1 );
legend('-DynamicLegend');
