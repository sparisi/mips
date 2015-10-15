function dam_plot_policies(policies, episodic)
% Plots some policies with single-dimensional state and action.
% Set 'episodic' to 1 if the policies are obtained with an episodic-based
% approach.
% The red policies are the learned ones. In blue, the real action according 
% to the problem constraints (min/max release).

n_pol = length(policies);
[~, policy] = dam_settings;
policy = policy.makeDeterministic;
env = dam_environment;

n_samples = 50;
x = linspace(0,160,n_samples);

y_free = zeros(n_samples,n_pol);
y_constr = zeros(n_samples,n_pol);
z = zeros(n_samples,n_pol);

for h = 1 : n_pol
    
    if episodic
        pol_high = policies(h).makeDeterministic;
        theta = pol_high.drawAction;
        policy.theta(1:length(theta)) = theta;
    else
        policy = policies(h).makeDeterministic;
    end
    
    a = policy.drawAction(x);
    y_free(:,h) = a;
    min_a = max(x - env.S_MIN_REL, 0);
    max_a = x;
    idx = min_a > a | max_a < a;
    a(idx) = max(min_a(idx), min(max_a(idx), a(idx)));
    y_constr(:,h) = a;

    z(:,h) = h*ones(n_samples,1);
    
end

figure; hold on
if n_pol == 1
    plot(x,y_constr,'-b');
    plot(x,y_free,'-.r','LineWidth',2);
else
    for i = 1 : n_pol
        plot3(z(:,i),x(:,i),y_constr(:,i),'-b');
        plot3(z(:,i),x(:,i),y_free(:,i),'-.r','LineWidth',2);
        ylabel state
        zlabel action
        set(gca,'XTick',[]);
    end
    view(-53,32)
end
legend('Constrained','Unconstrained')

grid on