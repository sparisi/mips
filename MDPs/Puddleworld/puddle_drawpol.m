function puddle_drawpol(policy)

mdpvars = puddle_mdpvariables();
nactions = length(mdpvars.action_list);
prob = [];
points = [];

for x = 0 : 0.05 : 1
    for y = 0 : 0.05 : 1
        for i = 1 : nactions
            action = mdpvars.action_list(i);
            p(i) = policy.evaluate([x; y], action);
        end
        points = [points; x y];
        prob = [prob; p];
    end
end

for i = 1 : nactions
%     scatter3(points(:,1), points(:,2), prob(:,i),'.')
    surfFromScatter(points(:,1), points(:,2), prob(:,i), 0.5)
    title(['Action ' num2str(i)])
end