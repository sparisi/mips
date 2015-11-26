function mce_plot(state, action)

persistent figureHandle agentHandle

if isempty(findobj('type','figure','name','MCarE Plot'))
    figureHandle = figure();
    figureHandle.Name = 'MCarE Plot';
    hold all
    
    env = mce_environment;
    
    xEnv = linspace(env.xLB,env.xUB,100)';
    yEnv = hill(xEnv);
    
    baseline = min(yEnv) - 0.1;
    
    fill([xEnv(1); xEnv; xEnv(end)],[baseline, yEnv, baseline],'g') % Hill
    plot(xEnv(end),yEnv(end),'diamond','MarkerSize',12,'MarkerFaceColor','m') % Goal
    plot(xEnv,yEnv,'b','LineWidth',2); % Road
    
    agentHandle = plot(-0.5,hill(-0.5),'ro','MarkerSize',8,'MarkerFaceColor','r'); % Car

    axis([env.xLB,env.xUB,baseline,max(yEnv)])
end

if nargin == 0
    return
end

agentHandle.XData = state(1);
agentHandle.YData = hill(state(1));

end

%%
function y = hill(x)
% Equation of the hill

idx = x < 0;
y(idx) = x(idx).^2 + x(idx);
y(~idx) = x(~idx) ./ sqrt(1 + 5*x(~idx).^2);

end
