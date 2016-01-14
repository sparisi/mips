classdef (Abstract) PolicyDiscrete < Policy
% POLICYDISCRETE Generic class for policies with discrete actions.
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        action_list
    end
    
    methods
        
        %% Value functions
        Q = obj.qFunction(States,Actions);
        
        function V = vFunction(obj, States)
            prob_list = obj.distribution(States);
            Q = obj.qFunction(States);
            V = mean(Q .* prob_list);
        end        
        
        %% Distribution functions
        prob_list = obj.distribution(States);

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
            idx = [1 : nlist : naction*nlist] + idx - 1;
            prob_list = prob_list(:);
            probability = prob_list(idx)';
        end
        
        function Actions = drawAction(obj, States)
            nstates = size(States,2);
            npolicy = numel(obj);
            if npolicy == 1 % Draw one action for each state
                prob_list = obj.distribution(States);
                [idsample, Actions] = find(mnrnd(ones(nstates,1), prob_list'));
                [~, idx] = sort(idsample);
                Actions = Actions(idx)';
            elseif npolicy == nstates % Draw one action for each pair (policy,state)
                Actions = zeros(1,nstates);
                for i = 1 : nstates
                    prob_list = obj(i).distribution(States(:,i));
                    Actions(i) = find(mnrnd(1, prob_list));
                end
            else
                error('Number of states and policies must be consistent.')
            end
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

            start_idx = (Actions-1)*dphi+1 + [0:nstates-1]*dphi*nactions; % Column start linear indices
            all_idx = bsxfun(@plus,start_idx,[0:dphi-1]'); % All linear indices
            phiSA = zeros(dphi*nactions,nstates); % Initialize output array with zeros
            phiSA(all_idx) = Phi; % Insert values from phi into output array
            phiSA(end-dphi+1:end,:) = []; % Last action does not have any explicit preference
        end
        
        %% Adds the constant feature 1 to the basis function
        function phi1 = basis1(obj, States)
            phi1 = [ones(1,size(States,2)); obj.basis(States)];
        end
        
        %% Plotting
        function plotDeterministic(obj, xmin, xmax, ymin, ymax, fig)
        % Plot most probable action for 2D states
            assert(xmin < xmax, 'X upper bound cannot be lower than lower bound.')
            assert(ymin < ymax, 'Y upper bound cannot be lower than lower bound.')

            if nargin == 5, figure, else figure(fig), end
            
            nactions = length(obj.action_list);
            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            
            actions = obj.makeDeterministic.drawAction([X(:)';Y(:)']);
            Z = reshape(actions,step,step);
            contourf(X,Y,Z)
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
            drawnow
        end
        
        function plotActions(obj, xmin, xmax, ymin, ymax, fig)
        % Plot actions distribution for 2D states
            assert(xmin < xmax, 'X upper bound cannot be lower than lower bound.')
            assert(ymin < ymax, 'Y upper bound cannot be lower than lower bound.')
            
            if nargin == 5, figure, else figure(fig), end

            nactions = length(obj.action_list);
            n = floor(sqrt(nactions));
            m = ceil(nactions/n);
            
            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            
            probs = obj.distribution([X(:)';Y(:)']);
            for i = obj.action_list
                subplot(n,m,i,'align')
                Z = reshape(probs(i,:),step,step);
                contourf(X,Y,Z)
                title(['Action ' num2str(i)])
                xlabel x
                ylabel y
            end
            drawnow
        end
        
        function plotQ(obj, xmin, xmax, ymin, ymax, fig)
        % Plot Q-function for 2D states
            assert(xmin < xmax, 'X upper bound cannot be lower than lower bound.')
            assert(ymin < ymax, 'Y upper bound cannot be lower than lower bound.')
            
            if nargin == 5, figure, else figure(fig), end

            nactions = length(obj.action_list);
            n = floor(sqrt(nactions));
            m = ceil(nactions/n);
            
            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            
            Q = obj.qFunction([X(:)';Y(:)']);
            for i = obj.action_list
                subplot(n,m,i,'align')
                Z = reshape(Q(i,:),step,step);
                contourf(X,Y,Z)
                title(['Q(s,' num2str(i) ')'])
                xlabel x
                ylabel y
            end
            drawnow
        end
        
        function plotV(obj, xmin, xmax, ymin, ymax, fig)
        % Plot V-function for 2D states
            assert(xmin < xmax, 'X upper bound cannot be lower than lower bound.')
            assert(ymin < ymax, 'Y upper bound cannot be lower than lower bound.')
            
            if nargin == 5, figure, else figure(fig), end
            
            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            
            Z = obj.vFunction([X(:)';Y(:)']);
            Z = reshape(Z,step,step);
            contourf(X,Y,Z)
            title('V-function')
            xlabel x
            ylabel y
            drawnow
        end
        
    end
    
end
