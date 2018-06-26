classdef MDP_avg < MDP
% Introduces state resets to learn average reward MDPs as if they were
% discounted.
%
% =========================================================================
% REFERENCE
% H van Hoof, G Neumann, J Peters
% Non-parametric Policy Search with Limited Information Loss (2017)

    properties (GetAccess = 'public', SetAccess = 'private')
        reset_prob
        mdp
    end

    properties
        dstate
        daction
        dreward
        isAveraged = 1;
        gamma
        stateLB
        stateUB
        rewardLB
        rewardUB
        actionLB
        actionUB
    end
    
    methods
        function obj = MDP_avg(mdp, reset_prob)
            obj.mdp = mdp;
            obj.reset_prob = reset_prob;
            obj.dstate = mdp.dstate;
            obj.daction = mdp.daction;
            obj.dreward = mdp.dreward;
            obj.gamma = 1 - reset_prob;
            obj.stateLB = mdp.stateLB;
            obj.stateUB = mdp.stateUB;
            obj.rewardLB = mdp.rewardLB;
            obj.rewardUB = mdp.rewardUB;
            obj.actionLB = mdp.actionLB;
            obj.actionUB = mdp.actionUB;
        end
        
        function state = initstate(obj, varargin)
            state = obj.mdp.initstate(varargin{:});
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            [nextstate, reward, absorb] = obj.mdp.simulator(state, action);
            idx = ~absorb;
            idx2 = rand(size(idx)) < obj.reset_prob;
            absorb(idx(idx2)) = true;
        end
        
        function showplot(obj)
            obj.mdp.showplot();
        end
        
        function closeplot(obj)
            obj.mdp.closeplot();
        end
        
        function plotepisode(obj, varargin)
            obj.mdp.plotepisode(varargin{:});
        end
        
        function plot_trajectories(obj, varargin)
            obj.mdp.plot_trajectories(varargin{:});
        end        
    end
    
end