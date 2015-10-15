function deep_animation(episode)

deep_plotenv
nexts = episode.nexts;

% Convert coordinates from cartesian to matrix
x = nexts(2,:); % (X,Y) -> (Y,X)
y = nexts(1,:);
nrows = size(deep_environment);
convertY = -(-nrows:-1); % Cartesian coord -> Matrix coord
nexts = [x; convertY(y)];

plot(1,nrows,'ro','MarkerSize',8,'MarkerFaceColor','r'); % Submarine init
pause(0.5)

for k = 1 : size(nexts,2)
    clf, hold all
    deep_plotenv
    plot(nexts(1,k),nexts(2,k),'ro','MarkerSize',8,'MarkerFaceColor','r'); % Submarine
    
    switch episode.a(k)
        case 1, a = 'Left';
        case 2, a = 'Right';
        case 3, a = 'Up';
        case 4, a = 'Down';
    end
        
    title(a)
    drawnow
    pause(0.5)
end

end
