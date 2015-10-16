function puddle2_animation(episode)

puddle2_plotenv
nexts = episode.nexts;
plot(episode.s(1,1),episode.s(2,1),'ro','MarkerSize',8,'MarkerFaceColor','r'); % Agent
pause(0.5)

for k = 1 : size(nexts,2)
    clf, hold all
    puddle2_plotenv
    plot(nexts(1,k),nexts(2,k),'ro','MarkerSize',8,'MarkerFaceColor','r'); % Agent
    
    switch episode.a(k)
        case 1, a = 'Left';
        case 2, a = 'Right';
        case 3, a = 'Up';
        case 4, a = 'Down';
    end
        
    title(['S: [' num2str(nexts(1,k)) ', ' num2str(nexts(2,k)) '], A: ' a])
    drawnow
    pause(0.5)
end

end
