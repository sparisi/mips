close all
clear all

figure
filename = 'animation.gif';
dt = 0.1;
%hold all % If you want to overlap plots

data = rand(100,10); % Use your data here
n = size(data,2);
F(n) = struct('cdata',[],'colormap',[]);

for i = 1 : n
    plot(data(:,i))
    F(i) = getframe;
    [X, map] = rgb2ind(frame2im(F(i)),256);
    
    if i == 1
        imwrite(X, map, filename, 'Loopcount', inf, 'DelayTime', dt)
    else
        imwrite(X, map, filename, 'WriteMode', 'Append', 'DelayTime', dt)
    end
end

%movie(F)