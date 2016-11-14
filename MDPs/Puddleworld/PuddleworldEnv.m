classdef PuddleworldEnv < MDP
% REFERENCE
% J A Boyan and A W Moore
% Generalization in Reinforcement Learning: Safely Approximating the Value 
% Function (1995)
    
    %% Properties
    properties
        % Environment variables
        goal = [1 1]';
        step = 0.05;
        p1 = [0.1 0.75; % Centers of the first puddle
            0.45 0.75];
        p2 = [0.45 0.4; % Centers of the second puddle
            0.45 0.8];
        radius = 0.1; % Radius of the puddles
    end
    
    methods
        
        function nextstate = transition(obj, state, action)
            nextstate = state + action + mymvnrnd([0;0],0.01^2*eye(2),size(state,2));
            nextstate = bsxfun(@max, bsxfun(@min,nextstate,obj.stateUB), obj.stateLB);
        end
        
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
            
            penalty = zeros(1,nstates);
            min_distance_from_puddle = min([d1; d2]);
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
            
            % Rectangles
            patch([0.1 0.45 0.45 0.1], [0.65 0.65 0.85 0.85], grey, 'EdgeAlpha', 0)
            patch([0.35 0.55 0.55 0.35], [0.4 0.4 0.8 0.8], grey, 'EdgeAlpha', 0)
            
            % Triangle
            x = [0.9, 1.0, 1.0];
            y = [1.0, 0.9, 1.0];
            fill(x, y, 'r')
            
            axis([0 1 0 1])
            box on
            axis square

            % Agent
            obj.handleAgent = plot(0.1,0.2,'ko','MarkerSize',12,'MarkerFaceColor','b');
        end
        
        function updateplot(obj, state)
            obj.handleAgent.XData = state(1);
            obj.handleAgent.YData = state(2);
            drawnow limitrate
        end
        
    end
    
end