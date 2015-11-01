function plotreturn(return_history)
% PLOTRETURN Plots mean and std of the average return.
%
%    INPUT
%     - return_history : N-by-I matrix, where N is the number of samples
%                        collected per iteration and I is the number of 
%                        iterations

shadedErrorBar(1:size(return_history,2), ...
    mean(return_history), ...
    std(return_history), ...
    {'LineWidth', 2'}, 0.1);
xlabel('Iterations')
ylabel('Average return')