classdef CollectWallSparse < CollectWall
% Like COLLECTWALL, but the reward based on the distance is not given anymore.
% Only the positive reward for collecting and for delivering the coin is given, 
% plus the penalty for hitting the wall or the environment boundaries.
% The exploration is therefore very hard, because initially the agent will
% often hit the wall and "be scared" of going to new places.
% To make it even more challenging, fix the initial position.
    
    methods
        
        %% Simulator
%         function state = init(obj, n)
%             agent = rand(2,n)+1; % Random in the upper right area, far from the coin
%             hole_center = rand(1,n);
%             has_coin = false(1,n);
%             state = [agent; hole_center; has_coin];
%         end
        
        function [nextstate, reward, absorb] = simulator(obj, state, action)
            nstates = size(state,2);

            % Init reward and absorb
            absorb = false(1, nstates);
            reward = zeros(1, nstates);
            
            % Parse state
            agent = state(1:2,:);
            hole_center = state(3,:);
            has_coin = state(4,:);

            % Parse action
            action = bsxfun(@times, action, 1 ./ max(matrixnorms(action,2),1));
            
            % Move agent
            next_agent = agent + action * obj.step;
            out_of_bounds = any(bsxfun(@le, next_agent, obj.stateLB(1:2)) | bsxfun(@ge, next_agent, obj.stateUB(1:2)));
            crossing = ( (agent(2,:) < obj.hole_y) & (next_agent(2,:) > obj.hole_y) ) ...
                | ( (agent(2,:) > obj.hole_y) & (next_agent(2,:) < obj.hole_y) );
            blocked = crossing & ~( agent(1,:) > hole_center - obj.hole_width/2 & agent(1,:) < hole_center + obj.hole_width/2 );
            next_agent = bsxfun(@max, bsxfun(@min,next_agent,obj.stateUB(1:2)), obj.stateLB(1:2)); % Put agent back in bounds
            next_agent(:,blocked) = agent(:,blocked);
            
            % Move barrier
            next_barrier = mod(hole_center + obj.step_hole, 1);

            % Check coin delivered
            delivered = has_coin & obj.at_pos(next_agent, obj.goal);
            absorb(delivered) = true;
            reward(delivered) = 1;
            
            % Check coin picked
            next_has_coin = has_coin;
            next_has_coin(obj.at_pos(next_agent, obj.coin)) = true;
            reward(next_has_coin & ~has_coin) = 1;
            
            % Penalties
            reward(out_of_bounds | blocked) = reward(out_of_bounds | blocked) - 1;

            % Create next state
            nextstate = [next_agent
                next_barrier
                next_has_coin];

            if obj.realtimeplot, obj.updateplot(nextstate), end
        end
        
    end
        
end
