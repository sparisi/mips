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
        
        %% Basis depending on the action
        function phiSA = duplicatebasis(obj, Phi, Actions)
            dphi = size(Phi,1);
            nstates = size(Phi,2);
            nactions = length(obj.action_list);
            [found,~] = (ismember(Actions,obj.action_list));
            assert(min(found) == 1);
            assert(isrow(Actions))
            assert(length(Actions) == nstates)

            idx_start = (Actions-1)*dphi+1 + (0:nstates-1)*dphi*nactions; % Starting linear indices
            idx = bsxfun(@plus,idx_start,(0:dphi-1)'); % All linear indices
            phiSA = zeros(dphi*nactions,nstates); % Initialize output array with zeros
            phiSA(idx) = Phi;
            phiSA = phiSA(1:length(obj.theta),:); % Some policies do not have any explicit preference on the last action
        end
        
        %% Adds the constant feature 1 to the basis function
        function phi1 = basis1(obj, States)
            phi1 = [ones(1,size(States,2)); obj.basis(States)];
        end
        
        %% Plotting
        function fig = plotGreedy(obj, LB, UB)
        % Plot the most probable action for 2D states.
            xmin = LB(1);
            ymin = LB(2);
            xmax = UB(1);
            ymax = UB(2);
            assert(xmin < xmax, 'X upper bound cannot be lower than lower bound.')
            assert(ymin < ymax, 'Y upper bound cannot be lower than lower bound.')

            nactions = length(obj.action_list);
            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            actions = obj.makeDeterministic.drawAction([X(:)';Y(:)']);
            Z = reshape(actions,step,step);
            
            fig = findobj('type','figure','name','Greedy');
            if isempty(fig)
                fig = figure();
                fig.Name = 'Greedy';
                surf(X,Y,Z,'EdgeColor','none')
                view(0,90)
                axis([xmin,xmax,ymin,ymax])
                title('Deterministic actions')
                xlabel x
                ylabel y
                cmap = jet(nactions);
                nonzeroActions = ismember(obj.action_list,actions);
                cmap = cmap(nonzeroActions,:);
                colormap(cmap)
                labels = num2cell(obj.action_list(nonzeroActions));
                labels = cellfun(@num2str,labels,'uni',0);
                lcolorbar(labels);
            else
                h = fig.Children(2).Children;
                h.XData = X;
                h.YData = Y;
                h.ZData = Z;
            end
            drawnow limitrate
        end
        
        function fig = plotQ(obj, LB, UB)
        % Plot Q-function for 2D states.
            xmin = LB(1);
            ymin = LB(2);
            xmax = UB(1);
            ymax = UB(2);
            assert(xmin < xmax, 'X upper bound cannot be lower than lower bound.')
            assert(ymin < ymax, 'Y upper bound cannot be lower than lower bound.')
            
            nactions = length(obj.action_list);
            n = floor(sqrt(nactions));
            m = ceil(nactions/n);
            
            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            Q = obj.qFunction([X(:)';Y(:)']);

            fig = findobj('type','figure','name','Q-function');
            if isempty(fig)
                fig = figure();
                fig.Name = 'Q-function';
                for i = obj.action_list
                    subplot(n,m,i,'align')
                    Z = reshape(Q(i,:),step,step);
                    contourf(X,Y,Z)
                    title(['Q(s,' num2str(i) ')'])
                    xlabel x
                    ylabel y
                end
            else
                for i = obj.action_list
                    h = fig.Children(i).Children;
                    Z = reshape(Q(i,:),step,step);
                    h.XData = X;
                    h.YData = Y;
                    h.ZData = Z;
                end
            end
            drawnow limitrate
        end
        
        function fig = plotV(obj, LB, UB)
        % Plot V-function for 2D states.
            xmin = LB(1);
            ymin = LB(2);
            xmax = UB(1);
            ymax = UB(2);
            assert(xmin < xmax, 'X upper bound cannot be lower than lower bound.')
            assert(ymin < ymax, 'Y upper bound cannot be lower than lower bound.')

            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            Z = obj.vFunction([X(:)';Y(:)']);
            Z = reshape(Z,step,step);
            
            fig = findobj('type','figure','name','V-function');
            if isempty(fig)
                fig = figure();
                fig.Name = 'V-function';
                contourf(X,Y,Z);
                title('V-function')
                xlabel x
                ylabel y
            else
                h = fig.Children.Children;
                h.XData = X;
                h.YData = Y;
                h.ZData = Z;
            end
            drawnow limitrate
        end
        
    end
    
end
