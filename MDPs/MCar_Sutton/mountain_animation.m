% Animated simulation.
% It needs a dataset 'ds'.

x = ds.s;
x = x(1,:);
x = x';
l = length(x);
tmp = ds.nexts;
xf = tmp(1,l);
x = [x;xf];
l = l+1;
y = ones(l,1);
x1 = [-1.2; -1.2];
y1 = [0; 10];
x2 = [0.6; 0.6];

for k = 1 : l
    plot(x(k),y(k),'bo')
    hold on; plot(x1,y1)
    hold on; plot(x2,y1)
    axis equal
    hold off
    M(k) = getframe;
end
