function plothistory(sample_history)
% PLOTHISTORY Given some samples collected during a learning process, it 
% plots their mean and std along time.
%
%    INPUT
%     - sample_history : N-by-I matrix, where N is the number of samples
%                        collected per iteration and I is the number of 
%                        iterations

shadedErrorBar(1:size(sample_history,2), ...
    mean(sample_history), ...
    std(sample_history), ...
    {'LineWidth', 2'}, 0.1);
xlabel('Iterations')
ylabel('Average value')