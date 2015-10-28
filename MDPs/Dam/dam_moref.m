function [front, weights, utopia, antiutopia] = dam_moref( outPlot )
% Returns reference frontiers, weights, utopia and antiutopia points for
% the problem. If OUTPLOT = 1, then it also plots the reference frontier.

n_obj = dam_settings;

if n_obj == 2
    
    front = dlmread('dam2_ref.dat');
    weights = dlmread('dam2_w.dat');
    utopia = [-0.5, -9];
    antiutopia = [-2.5, -11];
    
    if outPlot
        hold on
        plotfront(front,'s');
        xlabel 'Flooding'
        ylabel 'Water Demand'
        hold off
    end

elseif n_obj == 3
    
    front = dlmread('dam3_ref.dat');
    weights = dlmread('dam3_w.dat');
    utopia = [-0.5, -9, -0.001];
    antiutopia = [-65, -12, -0.7];
    
    if outPlot
        hold on
        plotfront(front,'o');
        xlabel 'Flooding'
        ylabel 'Water Demand'
        zlabel 'Hydroelectric Demand'
        hold off
    end
    
else
    
    front = dlmread('dam4_ref.dat');
    weights = dlmread('dam4_w.dat');
    utopia = [-0.5, -9, -0.001, -9];
    antiutopia = [-65, -12, -0.7, -12];
    
    if outPlot
        warning('Cannot display a 4-dimensional frontier.')
    end

end
