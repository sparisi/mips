classdef NLinkEnv < MDP
% REFERENCE
% T Yoshikawa
% Foundations of Robotics: Analysis and Control (1990)
    
    methods
        %% Simulator
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            original_action = action;
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);
            nextstate = obj.dynamics(state,action);
            reward = obj.reward_joint(original_action, nextstate);
            absorb = false(1,size(state,2));
            if obj.realtimeplot, obj.updateplot(nextstate); end
        end
        
        %% Rewards
        function reward = reward_task(obj, action, nextstate)
            X = obj.getJointsInTaskSpace(nextstate);
            endEffector = X(5:6,:);
            distance2 = sum(bsxfun(@minus,endEffector,obj.x).^2,1);
            reward = exp( -distance2 / (2*(1^2)) ) ... % The closer to the goal, the higher the reward
                - 0.05 * sum(abs(action),1); % Penalty on the action
        end
            
        function reward = reward_joint(obj, action, nextstate)
            distance = abs(angdiff(nextstate(1:2:end,:),obj.q_des,'rad'));
            penalty_dist = -matrixnorms(distance,2).^2;
%             penalty_dist = exp( -matrixnorms(distance,2).^2 );
            penalty_action = - 0.05 * sum(abs(action),1);
            reward = penalty_dist + penalty_action;
        end
    end        

    %% Plotting
    methods(Hidden = true)
        
        function initplot(obj)
            obj.handleEnv = figure(); hold all
            
            line(sum(obj.lengths)*[-1.1 1.1], [0 0], 'LineStyle', '--');
            axis(sum(obj.lengths)*[-1.1 1.1 -1.1 1.1]);

            % Agent handle
            lw = 4.0;
            colors = {[0.1 0.1 0.4], [0.4 0.4 0.8]};
            for i = 1 : 2 : length(obj.lengths)*2
                obj.handleAgent{i} = line([0 0], [0, 0], 'linewidth', lw, 'color', colors{mod((i+1)/2,2)+1});
                obj.handleAgent{i+1} = rectangle('Position',[0,0,0,0],'Curvature',[1,1],'FaceColor',colors{mod((i+1)/2,2)+1});
            end
        end
        
        function updateplot(obj, state)
            r = 0.1;
            X = obj.getJointsInTaskSpace(state);
            X = [ zeros(4,size(X,2)); X ];
            X(4:4:end,:) = []; % Remove velocities
            X(3:3:end,:) = [];
            for i = 1 : 2 : size(X,1) - 2
                obj.handleAgent{i}.XData = [X(i,1) X(i+2,1)];
                obj.handleAgent{i}.YData = [X(i+1,1), X(i+3,1)];
                obj.handleAgent{i+1}.Position = [X(i+2,1)-r,X(i+3,1)-r,2*r,2*r];
            end
            drawnow limitrate
        end
        
        function [pixels, clims, cmap] = render(obj, state)
            n_links = length(obj.lengths);
            meters_to_pixels = 5;
            tot_size = meters_to_pixels*n_links*2+2;
            
            if nargin == 1, pixels = tot_size^2; return, end
            
            n = size(state,2);
            pixels = zeros(tot_size,tot_size,n);
            
            X = obj.getJointsInTaskSpace(state) * meters_to_pixels;
            X = [ zeros(4,n); X ];
            X(4:4:end,:) = []; % Remove velocities
            X(3:3:end,:) = [];
            
            x = X(1:2:end-2,:)';
            y = X(2:2:end-2,:)';
            from = [x(:); y(:)]';
            x = X(3:2:end,:)';
            y = X(4:2:end,:)';
            to = [x(:); y(:)]';
            rows_cols = ceil( linspaceNDim(from(:)',to(:)',meters_to_pixels+1) + tot_size/2 );
            rows = rows_cols(1:end/2,:);
            cols = rows_cols(end/2+1:end,:);
            cols = -cols + tot_size;
            pages = repmat((1:n)',n_links,meters_to_pixels+1);
            pixels(sub2ind(size(pixels), cols, rows, pages)) = 1;
            
            clims = [min(min(min(pixels))), max(max(max(pixels)))];
            cmap = parula;
        end
        
    end
    
end