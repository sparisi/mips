classdef (Abstract) PolicyDiscrete < Policy
% POLICYDISCRETE Generic class for policies with discrete actions.
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        action_list
    end
    
    methods
        
        %% Value functions
        Q = qFunction(obj, States, Actions);
        
        function V = vFunction(obj, States)
            prob_list = obj.distribution(States);
            Q = obj.qFunction(States);
            V = sum(Q .* prob_list);
        end
        
        %% Distribution functions
        prob_list = distribution(obj, States);

        function probability = evaluate(obj, States, Actions)
            % Evaluate pairs (state, action)
            assert(length(Actions) == size(States,2), ...
                'The number of states and actions must be the same.');
            [found,idx] = (ismember(Actions,obj.action_list));
            assert(min(found) == 1, 'Unknown action.');
            
            % Get action probability
            prob_list = obj.distribution(States);
            nlist = length(obj.action_list);
            naction = length(Actions);
            idx = (1 : nlist : naction*nlist) + idx - 1;
            prob_list = prob_list(:);
            probability = prob_list(idx)';
        end
        
        function Actions = drawAction(obj, States)
            prob_list = obj.distribution(States);
            Actions = mymnrnd(prob_list,size(States,2)); % Draw one action for each state
        end
        
        function S = entropy(obj, States)
            prob_list = obj.distribution(States);
            idx = isinf(prob_list) | isnan(prob_list) | prob_list == 0;
            prob_list(idx) = 1; % ignore them -> log(1)*1 = 0
            S = -sum(prob_list.*log2(prob_list),1) / log2(length(obj.action_list));
            S = mean(S);
        end
        
        %% Plotting
        function plotGreedy(obj, LB, UB)
        % Plot the most probable action for 2D states.
            nactions = length(obj.action_list);
            step = 10;
            xnodes = linspace(LB(1),UB(1),step);
            ynodes = linspace(LB(2),UB(2),step);
            [X, Y] = meshgrid(xnodes,ynodes);
            A = obj.makeDeterministic.drawAction([X(:)';Y(:)']);
            A = reshape(A,step,step);
            
            fig = findobj('type','figure','name','Greedy Policy');
            if isempty(fig)
                figure('Name','Greedy Policy');
                surf(X,Y,A,'LineStyle','None','CDataMapping','direct')
                view(0,90)
                colormap(parula(nactions))
                axis([LB(1),UB(1),LB(2),UB(2)])
                lcolorbar(1:nactions)
            else
                h = fig.Children(2).Children;
                h.XData = X;
                h.YData = Y;
                h.ZData = A;
            end
            drawnow limitrate
        end
        
        function plotQ(obj, LB, UB)
        % Plot Q-function for 2D states.
            step = 10;
            x = linspace(LB(1),UB(1),step);
            y = linspace(LB(2),UB(2),step);
            [X, Y] = meshgrid(x,y);
            Q = obj.qFunction([X(:)';Y(:)']);
            subcontourf('Q-function',X,Y,Q)
        end
        
        function plotV(obj, LB, UB)
        % Plot V-function for 2D states.
            step = 10;
            x = linspace(LB(1),UB(1),step);
            y = linspace(LB(2),UB(2),step);
            [X, Y] = meshgrid(x,y);
            V = obj.vFunction([X(:)';Y(:)']);
            subcontourf('V-function',X,Y,V)
        end
        
    end
    
end
