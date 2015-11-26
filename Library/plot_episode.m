function plot_episode(domain, episode, pausetime)

if nargin == 2, pausetime = 0; end
    
feval([domain '_plot'], episode.s(:,1), episode.a(:,1)) % Plot init state

for i = 1 : size(episode.nexts,2) % Plot episode next states
    pause(pausetime)
    feval([domain '_plot'], episode.nexts(:,i), episode.a(:,i))
end
