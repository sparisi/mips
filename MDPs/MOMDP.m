classdef (Abstract) MOMDP < MDP
% MOMDP Like MDP, but with multiple rewards. It has variables and functions
% used for MORL, like utopia and antiutopia points and the Pareto frontier.
    
    properties (Abstract)
        utopia
        antiutopia
    end
    
    methods
        [front, weights] = truefront(obj);
        % Returns the true Pareto frontier (or a reference one) and a set of
        % weights if the frontier is obtained by weighted sum.
        
        fig = plotfront(obj, front, fig);
        % Plots a frontier and return the figure handle. To overlap two
        % frontiers, provide the same figure handle as input.
    end
    
end