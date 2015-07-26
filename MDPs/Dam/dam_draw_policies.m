function dam_draw_policies(policies, episodic)
% Plots some policies with single-dimensional state and action.
% Set 'episodic' to 1 if the policies are obtained with an episodic-based
% approach.
% The red policies are the learned ones. In blue, the real action according 
% to the problem constraints (min/max release).

n_pol = length(policies);
[~, policy] = dam_settings;
policy = policy.makeDeterministic;
env = dam_environment;
x = [];
y = [];
y2 = [];
z = [];

for h = 1 : n_pol
    
    a = [];
    a_real = [];
    s = [];
    
    if episodic
        pol_high = policies(h).makeDeterministic;
        theta = pol_high.drawAction;
        policy.theta(1:length(theta)) = theta;
    else
        policy = policies(h).makeDeterministic;
    end
    
    for i = 0 : 5 : 160
        
        s = [s; i];
        j = policy.drawAction(i);
        a = [a; j];
        
        min_a = max(i - env.S_MIN_REL, 0);
        max_a = i;
        if min_a > j || max_a < j
            j = max(min_a, min(max_a, j));
        end
        
        a_real = [a_real; j];
        
    end
    x = [x s];
    y = [y a];
    y2 = [y2 a_real];
    z = [z h*ones(33,1)];
    
end

figure; hold on
if n_pol == 1
    plot(x,y2,'-b');
    plot(x,y,'-.r','LineWidth',2);
else
    for i = 1 : n_pol
        plot3(z(:,i),x(:,i),y2(:,i),'-b');
        plot3(z(:,i),x(:,i),y(:,i),'-.r','LineWidth',2);
    end
    view(-53,32)
end
legend('Constrained','Unconstrained')

grid on