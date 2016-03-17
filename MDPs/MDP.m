classdef (Abstract) MDP < handle
% MDP Abstract class that defines the basic properties and methods of a 
% problem (number of states and actions, simulation and plotting, ...).
    
    properties (GetAccess = 'public', SetAccess = 'protected')
        realtimeplot = 0; % Flag to plot the environment at each timestep
        handleEnv         % Handle of the figure used for plotting
        handleAgent       % Handle of plotting elements inside handleEnv
    end
    
    properties (Abstract)
        dstate       % Size of the state space
        daction      % Size of the action space
        dreward      % Number of rewards
        isAveraged   % Is the reward averaged?
        gamma        % Discount factor
        
        % Upper/Lower Bounds for a tuple (state, action, reward)
        stateLB
        stateUB
        actionLB
        actionUB
        rewardLB
        rewardUB
        
        % For discrete MDPs, actions are encoded as integers (e.g., 1 ... 4
        % for left, right, up, down).
        % In this case, actionLB = 1 and actionUB = 4.
    end
    
    methods(Hidden = true)
        initplot(obj); % Initializes the environment and the agent figure handles.
        updateplot(obj, state); % Updates the figure handles.
    end
        
    methods
        [nextstate, reward, absorb] = simulator(obj, state, action);
        % Defines the state transition function.
        
        function showplot(obj)
        % Initializes the plotting procedure.
            obj.realtimeplot = 1;
            if isempty(obj.handleEnv), obj.initplot(); end
        end
        
        function closeplot(obj)
        % Closes the plots and stops the plotting procedure.
            obj.realtimeplot = 0;
            try close(obj.handleEnv), catch, end
            obj.handleEnv = [];
            obj.handleAgent = [];
        end
        
        function plotepisode(obj, episode, pausetime)
        % Plots the state of the MDP during an episode.
            if nargin == 2, pausetime = 0; end
            try close(obj.handleEnv), catch, end
            obj.initplot();
            obj.updateplot(episode.s(:,1));
            for i = 1 : size(episode.nexts,2)
                pause(pausetime)
                obj.updateplot(episode.nexts(:,i))
            end
        end
    end
    
end