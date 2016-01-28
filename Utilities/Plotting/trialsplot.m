% Script for plotting mean and avg of some data from several trials. Here
% it is assumed that your trials are saved in files named 'NAME1', 'NAME2',
% 'NAME3' ...

% close all
clear all

dataMatrix = [];
for i = 1 : 20
    try
        load(['NAME' num2str(i)])
        dataVector = J_history; % change to your desired data
        dataMatrix = [dataMatrix; dataVector];
    catch
    end
end

hold all
% dataMatrix = moving(dataMatrix',2)';
h = shadedErrorBar(1:size(dataMatrix,2), ...
    mean(dataMatrix,1), ...
    std(dataMatrix), ...
    {'LineWidth', 2}, 0.1);
