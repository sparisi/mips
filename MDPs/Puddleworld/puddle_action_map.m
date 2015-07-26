% Plots a map of the distribution of each action on the environment.

[~, policy] = puddle_settings; % change with the policy you want to plot

step = 0.05;

%left
map1 = zeros(1/step+1);
%right
map2 = zeros(1/step+1);
%up
map3 = zeros(1/step+1);
%down
map4 = zeros(1/step+1);
h = 0;

for i = 0 : step : 1
    h = h + 1;
    k = 0;
    for j = 0 : step : 1
        k = k + 1;
        distrib = zeros(4,1);
        state = [i; j];
        for action = 1 : 4
            distrib(action) = policy.evaluate(state,action);
        end
        map1(h,k) = distrib(1);
        map2(h,k) = distrib(2);
        map3(h,k) = distrib(3);
        map4(h,k) = distrib(4);
    end
end

map1 = rot90(map1);
map2 = rot90(map2);
map3 = rot90(map3);
map4 = rot90(map4);