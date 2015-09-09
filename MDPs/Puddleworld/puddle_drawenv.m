p1 = [0.1 0.75; % Centers of the first puddle
    0.45 0.75];
p2 = [0.45 0.4; % Centers of the second puddle
    0.45 0.8];
radius = 0.1;
ratio = 400; % Total number of cells

figure, hold all
count = 0;
reward = 0;

for x = 0 : 0.01 : 1
    for y = 0 : 0.01 : 1
        
        state = [x; y];
        
        if state(1) > p1(2,1)
            d1 = norm(state' - p1(2,:));
        elseif state(1) < p1(1,1)
            d1 = norm(state' - p1(1,:));
        else
            d1 = abs(state(2) - p1(1,2));
        end
        
        if state(2) > p2(2,2)
            d2 = norm(state' - p2(2,:));
        elseif state(2) < p2(1,2)
            d2 = norm(state' - p2(1,:));
        else
            d2 = abs(state(1) - p2(1,1));
        end
        
        min_distance_from_puddle = min(d1, d2);
        if min_distance_from_puddle <= radius
            plot(state(1),state(2),'k.')
            reward = reward - ratio * (radius - min_distance_from_puddle);
        end
        
        count = count + 1;
        
    end
end

axis([0 1 0 1])

meanRew = reward / count; % mean reward per cell
