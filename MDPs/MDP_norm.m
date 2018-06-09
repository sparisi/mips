classdef MDP_norm < MDP
% Wrapper over an MDP to normalize states and actions in [-1,1].
% Applicable only to MDPs with symmetric states and actions bounds.
% The MDP receives actions in [-1,1] and states in [-1,1], scales them to
% [-a_max,a_max] and [-s_max,s_max], and returns nextstates in [-1,1].
    
    properties
        mdp

        dstate
        daction
        dreward
        isAveraged
        gamma

        stateLB
        stateUB
        rewardLB
        rewardUB
        actionLB
        actionUB

        action_high
        state_high
    end
    
    methods
        
        function obj = MDP_norm(mdp)
           assert( ~any( isinf(mdp.stateLB) | isinf(mdp.stateUB) | ...
                   isinf(mdp.actionLB) | isinf(mdp.actionUB) ), ...
               'The MDP has unbounded state and/or action.' )
           assert( all( mdp.stateLB == -mdp.stateUB ) && ...
               all( mdp.actionLB == -mdp.actionUB ) , ...
               'Action and/or state bounds are not symmetric.')

           obj.mdp = mdp;
           obj.dstate = mdp.dstate;
           obj.daction = mdp.daction;
           obj.dreward = mdp.dreward;
           obj.isAveraged = mdp.isAveraged;
           obj.gamma = mdp.gamma;
           obj.stateLB = -ones(mdp.dstate,1);
           obj.stateUB = ones(mdp.dstate,1);
           obj.rewardLB = mdp.rewardLB;
           obj.rewardUB = mdp.rewardUB;
           obj.actionLB = ones(mdp.daction,1);
           obj.actionUB = ones(mdp.daction,1);
           
           obj.action_high = mdp.actionUB;
           obj.state_high = mdp.stateUB;
        end
        
        function state = initstate(obj,n)
            if nargin == 1, n = 1; end
            state = obj.mdp.initstate(n);
            state = bsxfun(@times, state, 1./obj.state_high);
        end
            
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            state = bsxfun(@times, state, obj.state_high);
            action = bsxfun(@times, action, obj.action_high);
            [nextstate, reward, absorb] = obj.mdp.simulator(state, action);
            nextstate = bsxfun(@times, nextstate, 1./obj.state_high);
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