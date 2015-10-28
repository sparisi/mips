function [front, weights, utopia, antiutopia] = lqr_moref( outPlot )
% Returns reference frontiers, weights, utopia and antiutopia points for
% the problem. If OUTPLOT = 1, then it also plots the reference frontier.
%
% NB: the frontiers have been obtained with a Gaussian policy with fixed
% identity covariance.

n_obj = lqr_settings;

if n_obj == 2 % starting from [10, 10]

    front = dlmread('lqr2_ref.dat');
    weights = dlmread('lqr2_w.dat');
    utopia = -150*ones(1,n_obj);
    antiutopia = -310*ones(1,n_obj);
    
    if outPlot
        hold on
        plotfront(front,'.');
        xlabel 'Obj 1'
        ylabel 'Obj 2'
        hold off
    end
    
elseif n_obj == 3 % starting from [10, 10, 10]
    
    front = dlmread('lqr3_ref.dat');
    weights = dlmread('lqr3_w.dat');
    utopia = -195*ones(1,n_obj);
    antiutopia = -360*ones(1,n_obj);
    
    if outPlot
        hold on
        plotfront(front,'o');
        xlabel 'Obj 1'
        ylabel 'Obj 2'
        zlabel 'Obj 3'
        hold off
    end
    
elseif n_obj == 5 % starting from [10, 10, 10, 10, 10]

    front = dlmread('lqr5_ref.dat');
    weights = dlmread('lqr5_w.dat');
    antiutopia = -436*ones(1,n_obj);
    utopia = -283*ones(1,n_obj);
 
else
    
    error('Frontier not available for the desired number of objectives.')
    
end

end

