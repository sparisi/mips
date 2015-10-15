function resource_animation(episode)

resource_plotenv
nexts = episode.nexts;

% Convert coordinates from cartesian to matrix
x = nexts(2,:); % (X,Y) -> (Y,X)
y = nexts(1,:);
[nrows, ncols] = size(resource_environment);
convertY = -(-nrows:-1); % Cartesian coord -> Matrix coord
nexts = [x; convertY(y); nexts(3:4,:)]; 

plot(3,1,'ro','MarkerSize',8,'MarkerFaceColor','r'); % Agent init
pause(0.1)

for k = 1 : size(nexts,2)
    clf, hold all
    resource_plotenv
    plot(nexts(1,k),nexts(2,k),'ro','MarkerSize',16,'MarkerFaceColor','r'); % Agent
    
    if nexts(3,k) % Carrying gems
        delete(hGems)
        plot(nexts(1,k)+0.3,nexts(2,k)+0.3,'kdiamond','MarkerSize',6,'MarkerFaceColor','m'); 
    end
    
    if nexts(4,k) % Carrying gold
        delete(hGold)
        plot(nexts(1,k)+0.3,nexts(2,k)+0.1,'ko','MarkerSize',6,'MarkerFaceColor','y'); 
    end
    
    if episode.r(1,k) % Fight lost
        patch([ncols 1 1 ncols], [nrows nrows 1 1], 'r', 'EdgeAlpha', 0)
        alpha(0.3)
    end
    
    if episode.r(2,k) || episode.r(3,k) % Successfully carried gold or gems home
        patch([ncols 1 1 ncols], [nrows nrows 1 1], 'g', 'EdgeAlpha', 0)
        alpha(0.3)
    end
    
    switch episode.a(k)
        case 1, a = 'Left';
        case 2, a = 'Right';
        case 3, a = 'Up';
        case 4, a = 'Down';
    end
    
    title(a)
    drawnow
    pause(0.1)
end

end
