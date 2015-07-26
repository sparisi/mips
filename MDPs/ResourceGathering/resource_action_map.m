% Plots a map of the distribution of each action on the environment.

[~, policy] = resource_settings; % change with the policy you want to plot

map1 = cell(5);
map2 = cell(5);

for i = 1 : 5
    for j = 1 : 5
        distrib = zeros(4,1);
        state = [i; j; 0; 0];
        for action = 1 : 4
            distrib(action) = policy.evaluate(state,action);
        end
        map1{i,j} = reshape(distrib,2,2)';
        
        state = [i; j; 0; 1];
        for action = 1 : 4
            distrib(action) = policy.evaluate(state,action);
        end
        map2{i,j} = reshape(distrib,2,2)';
    end
end

A1 = cell2mat(map1); % action map when the agent is carrying nothing
fprintfmat(A1, 2, 2);

A2 = cell2mat(map1); % action map when the agent is carrying a gem
fprintfmat(A2, 2, 2);
