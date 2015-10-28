function [front, weights, utopia, antiutopia] = puddle_moref( outPlot )
% Returns reference frontiers, weights, utopia and antiutopia points for
% the problem. If OUTPLOT = 1, then it also plots the reference frontier.

front = dlmread('puddle_ref.dat');
weights = [];
% warning('No weights are available for the Puddleworld domain.')
utopia = [];
antiutopia = [];

if outPlot
    hold on
    plotfront(front,'.');
    xlabel 'Steps'
    ylabel 'Puddle Penalty'
    hold off
end

end
