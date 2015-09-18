function drawDiscrete(policy)

nactions = length(policy.action_list);
prob = [];
points = [];

% Change the status range according to the problem
for x = 0 : 0.05 : 1
    for y = 0 : 0.05 : 1
        prob_list = policy.distribution([x;y])';
        points = [points; x y];
        prob = [prob; prob_list];
    end
end

for i = 1 : nactions
%     scatter3(points(:,1), points(:,2), prob(:,i),'.')
    surfFromScatter(points(:,1), points(:,2), prob(:,i), 0.5)
    title(['Action ' num2str(i)])
end