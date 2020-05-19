% This script iteratively loads data from .dat files located in FOLDER into 
% a matrix, and plots the column VAR_IDX. 

close all

folder = '/home/sparisi/tensorl/data-trial/ppo/Pendulum-v0/';
Files = dir(fullfile(folder,'*.dat'));

varbound = -inf;
var_idx = 6;

figure, hold all
for current_file = {Files.name}
    h = load([folder current_file{:}]);
    plot(max(h(:,var_idx), varbound))
end

legend({Files.name}, 'Interpreter', 'none')
