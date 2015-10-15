function mce_animation(episode)

env = mce_environment;

xEnv = linspace(env.xLB,env.xUB,100)';
yEnv = hill(xEnv);

xEp = episode.s(1,:);
yEp = hill(xEp);

baseline = min(yEnv) - 0.1;
figure
for k = 1 : length(yEp)
    clf, hold all
    fill([xEnv(1); xEnv; xEnv(end)],[baseline, yEnv, baseline],'g') % Hill
    plot(xEnv,yEnv,'b','LineWidth',2); % Road
    plot(xEnv(end),yEnv(end),'diamond','MarkerSize',12,'MarkerFaceColor','m') % Goal
    plot(xEp(k),yEp(k),'ro','MarkerSize',8,'MarkerFaceColor','r'); % Car

    axis([env.xLB,env.xUB,baseline,max(yEnv)])
    drawnow
    pause(0.05)
end

end

function y = hill(x)
% Equation of the hill
idx = x < 0;
y(idx) = x(idx).^2 + x(idx);
y(~idx) = x(~idx) ./ sqrt(1 + 5*x(~idx).^2);
end
