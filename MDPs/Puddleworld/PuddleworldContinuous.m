classdef PuddleworldContinuous < MDP
    
    %% Properties
    properties
        % Environment variables
        goal = [1 1]';
        p1 = [0.1 0.75; % Centers of the first puddle
            0.45 0.75];
        p2 = [0.45 0.4; % Centers of the second puddle
            0.45 0.8];
        p3 = [0.8 0.2;  % Centers of the third puddle
            0.8 0.5];
        p4 = [0.7 0.75; % Centers of the fourth puddle
            0.7 0.8];
        radius = 0.1; % Radius of the puddles
        
        % MDP variables
        dstate = 2;
        daction = 2;
        dreward = 1;
        isAveraged = 0;
        gamma = 1;

        % Bounds
        stateLB = [0 0]';
        stateUB = [1 1]';
        actionLB = -[0.05 0.05]';
        actionUB = [0.05 0.05]';
        rewardLB = -41;
        rewardUB = -1;
    end
    
    methods
        
        %% Simulator
        function state = initstate(obj, n)
            state = rand(2,n);
            if obj.realtimeplot, obj.showplot; obj.updateplot(state); end
        end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            % Bound the action
            action = bsxfun(@max, bsxfun(@min,action,obj.actionUB), obj.actionLB);

            % Transition function
            nextstate = state + action + normrnd(0,0.01,size(state));
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
            
            goaldistance = matrixnorms(bsxfun(@minus,nextstate,obj.goal),2);

            % Distance from the nearest edge of the puddle + Time penalty
            reward = obj.puddlepenalty(nextstate) - 1;
            
            % Terminal condition
            absorb = goaldistance <= 0.05;
            
            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
        %% Reward function
        function penalty = puddlepenalty(obj, state)
            factor = 400;
            nstates = size(state,2);
            
            idxR = state(1,:) > obj.p1(2,1); % states on the right of the right center
            idxL = state(1,:) < obj.p1(1,1); % on the left of the left center
            d1(:,idxR) = bsxfun(@minus,state(:,idxR),obj.p1(2,:)');
            d1(:,idxL) = bsxfun(@minus,state(:,idxL),obj.p1(1,:)');
            d1 = matrixnorms(d1,2);
            d1(~(idxR | idxL)) = abs(state(2,~(idxR | idxL)) - obj.p1(1,2)); % states in between the centers
            
            idxU = state(2,:) > obj.p2(2,2); % states above the upper center
            idxD = state(2,:) < obj.p2(1,2); % below the lower center
            d2(:,idxU) = bsxfun(@minus,state(:,idxU),obj.p2(2,:)');
            d2(:,idxD) = bsxfun(@minus,state(:,idxD),obj.p2(1,:)');
            d2 = matrixnorms(d2,2);
            d2(~(idxU | idxD)) = abs(state(1,~(idxU | idxD)) - obj.p2(1,1)); % states in between the centers
            
            idxU = state(2,:) > obj.p3(2,2); % states above the upper center
            idxD = state(2,:) < obj.p3(1,2); % below the lower center
            d3(:,idxU) = bsxfun(@minus,state(:,idxU),obj.p3(2,:)');
            d3(:,idxD) = bsxfun(@minus,state(:,idxD),obj.p3(1,:)');
            d3 = matrixnorms(d3,2);
            d3(~(idxU | idxD)) = abs(state(1,~(idxU | idxD)) - obj.p3(1,1)); % states in between the centers

            idxU = state(2,:) > obj.p4(2,2); % states above the upper center
            idxD = state(2,:) < obj.p4(1,2); % below the lower center
            d4(:,idxU) = bsxfun(@minus,state(:,idxU),obj.p4(2,:)');
            d4(:,idxD) = bsxfun(@minus,state(:,idxD),obj.p4(1,:)');
            d4 = matrixnorms(d4,2);
            d4(~(idxU | idxD)) = abs(state(1,~(idxU | idxD)) - obj.p4(1,1)); % states in between the centers

            penalty = zeros(1,nstates);
            min_distance_from_puddle = min([d1; d2; d3; d4]);
            idx = min_distance_from_puddle <= obj.radius;
            penalty(idx) = - factor * (obj.radius - min_distance_from_puddle(idx));
        end
        
    end
    
    methods(Hidden = true)
        
        %% Plotting
        function initplot(obj)
            obj.handleEnv = figure(); hold all
            grey = [0.4,0.4,0.4];
            
            % Circles
            rectangle('Position',[obj.p1(1,1)-obj.radius,obj.p1(1,2)-obj.radius,2*obj.radius,2*obj.radius], ...
                'Curvature',[1,1],'FaceColor',grey,'EdgeColor',grey);
            rectangle('Position',[obj.p1(2,1)-obj.radius,obj.p2(2,2)-obj.radius,2*obj.radius,2*obj.radius], ...
                'Curvature',[1,1],'FaceColor',grey,'EdgeColor',grey);
            rectangle('Position',[obj.p2(1,1)-obj.radius,obj.p2(1,2)-obj.radius,2*obj.radius,2*obj.radius], ...
                'Curvature',[1,1],'FaceColor',grey,'EdgeColor',grey);
            rectangle('Position',[obj.p2(2,1)-obj.radius,obj.p2(2,2)-obj.radius,2*obj.radius,2*obj.radius], ...
                'Curvature',[1,1],'FaceColor',grey,'EdgeColor',grey);
            rectangle('Position',[obj.p3(1,1)-obj.radius,obj.p3(1,2)-obj.radius,2*obj.radius,2*obj.radius], ...
                'Curvature',[1,1],'FaceColor',grey,'EdgeColor',grey);
            rectangle('Position',[obj.p3(2,1)-obj.radius,obj.p3(2,2)-obj.radius,2*obj.radius,2*obj.radius], ...
                'Curvature',[1,1],'FaceColor',grey,'EdgeColor',grey);
            rectangle('Position',[obj.p4(1,1)-obj.radius,obj.p4(1,2)-obj.radius,2*obj.radius,2*obj.radius], ...
                'Curvature',[1,1],'FaceColor',grey,'EdgeColor',grey);
            rectangle('Position',[obj.p4(2,1)-obj.radius,obj.p4(2,2)-obj.radius,2*obj.radius,2*obj.radius], ...
                'Curvature',[1,1],'FaceColor',grey,'EdgeColor',grey);
            
            % Rectangles
            patch([0.1 0.45 0.45 0.1], [0.65 0.65 0.85 0.85], grey, 'EdgeAlpha', 0)
            patch([0.35 0.55 0.55 0.35], [0.4 0.4 0.8 0.8], grey, 'EdgeAlpha', 0)
            patch([0.7 0.9 0.9 0.7], [0.2 0.2 0.5 0.5], grey, 'EdgeAlpha', 0)
            patch([0.6 0.8 0.8 0.6], [0.75 0.75 0.8 0.8], grey, 'EdgeAlpha', 0)
            
            % Triangle
            x = [0.95, 1.0, 1.0];
            y = [1.0, 0.95, 1.0];
            fill(x, y, 'r')
            
            axis([0 1 0 1])
            box on
            axis square

            % Agent
            obj.handleAgent = plot(0,0,'ro','MarkerSize',8,'MarkerFaceColor','r');
        end
        
        function updateplot(obj, state)
            obj.handleAgent.XData = state(1);
            obj.handleAgent.YData = state(2);
        end
        
    end
    
end