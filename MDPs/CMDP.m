classdef (Abstract) CMDP < MDP
% CMDP Are Contextual MDP. At the beginning of an episode, the context is 
% set and kept fixed until the end of the episode.
    
    properties (Abstract)
        dctx  % Size of the context
        ctxLB % Context bounds
        ctxUB
    end
    
    methods (Abstract)
        context = initctx(obj,n);
    end
    
    methods
        function plotepisode(obj, episode, context, pausetime)
        % Plots the state of the MDP during an episode for a given context.
            if nargin == 3, pausetime = 0.001; end
            try close(obj.handleEnv), catch, end
            obj.initplot(context);
            obj.updateplot(episode.s(:,1));
            for i = 1 : size(episode.nexts,2)
                pause(pausetime)
                obj.updateplot(episode.nexts(:,i))
                title(['Step ' num2str(i) ',   Reward ' strrep(mat2str(episode.r(:,i)'), ' ', ', ')])
            end
        end
    end        
    
end