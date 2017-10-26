% Compares results of different methods averaged over many trials. 
% You need to set the folder where the data is stored and the filename
% format (eg, alg1_1.mat, alg1_2.mat, ..., alg2_1.mat, ...)

close all
clear, clear global

%% Change entries according to your needs
folder = 'results/';
separator = '_';
filenames = {'alg1', 'alg2', 'alg3'};
variable = 'J_history';
variable = 'mean(J_history)';
variable = '-log(-mean(J_history))';

%% Plot
h = {};
for name = filenames
    
    counter = 1;
    dataMatrix = [];
    for trial = 1 : 999
        try
            load([folder name{:} separator num2str(trial)])
            dataMatrix(counter,:) = eval(variable);
            counter = counter + 1;
        catch
        end
    end
    % dataMatrix = moving(dataMatrix',2)';
    
    if ~isempty(dataMatrix)
        hold all
        tmp = shadedErrorBar( ...
            1:size(dataMatrix,2), ...
            mean(dataMatrix,1), ...
            std(dataMatrix), ...
            { 'LineWidth', 2, 'DisplayName', name{:} }, ...
            0.1 );
        h{end+1} = tmp.mainLine;
    end
    
end

legend([h{:}],filenames)

leg.Position = [0.2 0.7 0 0];