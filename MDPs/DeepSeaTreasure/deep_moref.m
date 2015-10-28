function [front, weights, utopia, antiutopia] = deep_moref( outPlot )
% Returns reference frontiers, weights, utopia and antiutopia points for
% the problem. If OUTPLOT = 1, then it also plots the reference frontier.

front = dlmread('deep_ref.dat');
weights = [];
% warning('No weights are available for the Deep Sea Treasure domain.')
utopia = [124, -1];
antiutopia = [0, -20];

if outPlot
    hold on
    plotfront(front,'s');
    xlabel 'Treasure'
    ylabel 'Time'
    hold off
end

end
