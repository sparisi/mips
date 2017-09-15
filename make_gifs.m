% Single plot

close all
clear all

filename = 'animation2';
dt = 1; % Pause between each frame
F = struct('cdata',[],'colormap',[]);

fh = figure('units','pixels','position',[100 100 1092 384]);
set(gcf,'color','w');
%hold all % If you want to overlap plots

n = 10;
data1 = rand(100,n); % Use your data here

cont = 1;
for i = 1 : n
    plot(data1(:,i),'bo')
    axis([0,100,0,1])
    
    ax = gca;
    ax.Units = 'pixels';
    pos = ax.Position;
    marg = 30;
    rect = [-marg, -marg, pos(3)+2*marg, pos(4)+2*marg];
    F(cont) = getframe(ax, rect);

    % Pass axis instead of figure to getframe in order to capture only the
    % inside of the plot. The margin can be used to get also axes ticks.
    
    [X, map] = rgb2ind(frame2im(F(cont)),256);
    cont = cont + 1;
    
    % Write gif
    if i == 1
        imwrite(X, map, [filename '.gif'], 'Loopcount', inf, 'DelayTime', dt)
    else
        imwrite(X, map, [filename '.gif'], 'WriteMode', 'Append', 'DelayTime', dt)
    end
end

% Write video
v = VideoWriter(filename);
v.FrameRate = 1/dt;
open(v)
writeVideo(v,F)
close(v)
