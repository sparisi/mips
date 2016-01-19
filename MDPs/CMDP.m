classdef (Abstract) CMDP < MDP
% CMDP Are Contextual MDP. At the beginning of an episode, the context is 
% set and kept fixed until the end of the episode.
    
    properties (Abstract)
        dctx  % Size of the context
        ctxLB % Context bounds
        ctxUB
    end
    
    methods (Abstract)
        context = getcontext(obj,n);
    end
    
end