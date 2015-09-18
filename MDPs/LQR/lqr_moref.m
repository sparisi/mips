function [front, weights, utopia, antiutopia] = lqr_moref( outPlot )
% Returns reference frontiers, weights, utopia and antiutopia points for
% the problem. If OUTPLOT = 1, then it also plots the reference frontier.

n_obj = lqr_settings;

if n_obj == 2 % starting from [10, 10]

    front = dlmread('lqr2_ref.dat');
    weights = dlmread('lqr2_w.dat');
    utopia = [-150, -150];
    antiutopia = [-310, -310];
    
    if outPlot
        hold on
        plot(front(:,1), front(:,2),'b.-')
        xlabel 'Obj 1'
        ylabel 'Obj 2'
        hold off
    end
    
elseif n_obj == 3 % starting from [10, 10, 10]
    
    front = dlmread('lqr3_ref.dat');
    weights = dlmread('lqr3_w.dat');
    utopia = [-195, -195, -195];
    antiutopia = [-360, -360, -360];
    
    if outPlot
        hold on
        scatter3(front(:,1), front(:,2), front(:,3), 'bo')
        xlabel 'Obj 1'
        ylabel 'Obj 2'
        zlabel 'Obj 3'
        hold off
    end
    
else
    
    error('Frontier not available for more than 3 objectives.')
    
end

end

