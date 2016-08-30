classdef (Abstract) MOMDP < MDP
% MOMDP Like MDP, but with multiple rewards. It has variables and functions
% used for MORL, like utopia and antiutopia points and the Pareto frontier.
    
    properties (Abstract)
        utopia
        antiutopia
    end
    
    methods
        % NB! Frontiers are returned / passed as [N x R] matrices, where N
        % is the number of points and R is the number of objectives.
        
        [front, weights] = truefront(obj);
        % Returns the true Pareto frontier (or a reference one) and a set of
        % weights if the frontier is obtained by weighted sum.
    end
    
    methods(Static)
        function fig = plotfront(front, varargin)
        % Plots a frontier and returns the figure handle.
            front = sortrows(front);
            if size(front,2) == 2
                fig = plot(front(:,1),front(:,2),varargin{:});
            elseif size(front,2) == 3
                fig = plot3(front(:,1),front(:,2),front(:,3),varargin{:});
                box on, grid on
            else
                warning('Cannot plot more than three objectives.')
            end
        end
    end
    
end