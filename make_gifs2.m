% Multiple subplots

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
data2 = exp(data1);

cont = 1;
for i = 1 : n
    subplot(1,2,1)
    plot(data1(:,i),'bo')
    axis([0,100,0,1])
    
    subplot(1,2,2)
    plot(data2(:,i),'r--o')
    axis([0,100,1,2])

    F(cont) = getframe(fh);
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
