classdef (Abstract) policy_discrete < policy
% POLICY_DISCRETE Generic class for policies with discrete actions.
    
    methods (Abstract)
        distribution(obj)
    end
    
    methods
        
        % In all methods, STATES are matrices S-by-N, where S is the size
        % of one state and N is the number of states.
        % Actions are always scalar values.

        function v = vfunction(obj, States)
            [probs, q] = obj.distribution(States);
            v = mean(q .* probs);
        end        
        
        function probability = evaluate(obj, States, action)
            % Assert that the action belongs to the known actions
            idx = find(obj.action_list == action);
            assert(length(idx) == 1);
            
            % Get action probability
            prob_list = obj.distribution(States);
            probability = prob_list(idx,:);
        end
        
        function Actions = drawAction(obj, States)
            nstates = size(States,2); % Draw one action for each state
            prob_list = obj.distribution(States);
            [~, Actions] = find(mnrnd(ones(nstates,1), prob_list'));
        end
        
        function obj = randomize(obj, factor)
            obj.theta = obj.theta ./ factor;
        end
        
        function S = entropy(obj, States)
            nstates = size(States,2);
            nactions = length(obj.action_list);
            prob_list = obj.distribution(States);

            S = zeros(1,nstates);
            for i = 1 : nactions
                % Usual checks for the entropy
                if ~max((isinf(prob_list(i,:)) | isnan(prob_list(i,:)) | prob_list(i,:) == 0))
                    S = S + (-prob_list(i,:).*log2(prob_list(i,:)));
                end
            end
            S = S / log2(nactions);
            S = mean(S);
        end
        
        %% PLOTTING
        %%% Plot actions distribution for 2D states
        function plot(obj, xmin, xmax, ymin, ymax, fig)
            assert(xmin < xmax)
            assert(ymin < ymax)
            
            scrsz = get(groot,'ScreenSize');
            if nargin == 5
                figure('Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])
            else
                set(fig,'Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])
                figure(fig)
            end

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
        end
        
        %%% Plot Q-function for 2D states
        function plotQ(obj, xmin, xmax, ymin, ymax, fig)
            assert(xmin < xmax)
            assert(ymin < ymax)
            
            scrsz = get(groot,'ScreenSize');
            if nargin == 5
                figure('Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])
            else
                set(fig,'Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])
                figure(fig)
            end

            nactions = length(obj.action_list);
            n = floor(sqrt(nactions));
            m = ceil(nactions/n);
            
            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            
            [~, Q] = obj.distribution([X(:)';Y(:)']);
            for i = obj.action_list
                subplot(n,m,i,'align')
                Z = reshape(Q(i,:),step,step);
                contourf(X,Y,Z)
                title(['Action ' num2str(i)])
                xlabel x
                ylabel y
            end
        end
        
        %%% Plot V-function for 2D states
        function plotV(obj, xmin, xmax, ymin, ymax, fig)
            assert(xmin < xmax)
            assert(ymin < ymax)
            
            if nargin == 5
                figure
            else
                figure(fig)
            end
            
            step = 30;
            xnodes = linspace(xmin,xmax,step);
            ynodes = linspace(ymin,ymax,step);
            [X, Y] = meshgrid(xnodes,ynodes);
            
            Z = obj.vfunction([X(:)';Y(:)']);
            Z = reshape(Z,step,step);
            contourf(X,Y,Z)
            title('V-function')
            xlabel x
            ylabel y
        end        
        
    end
    
end
