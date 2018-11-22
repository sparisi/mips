% Simple script to plot the trend of a specific variable VARNAME saved in 
% all .mat files located in a specific FOLDER.

close all

folder = 'data/';
Files = dir(fullfile(folder,'*.mat'));

varname = 'J_history';
varbound = -inf;

figure, hold all, title(varname, 'interpreter', 'none')
for current_file = {Files.name}
    load([folder current_file{:}], varname);
    try
        plot(max(eval(varname), varbound))
    catch
        warning(['Variable "' varname '" does not exist in file "' current_file{:} '".'])
    end
end

legend({Files.name}, 'Interpreter', 'none')
