function make_gif_simulation(filename, mdp, policy, steps, pausetime, render)
% MAKE_GIF_SIMULATION Runs an episode and shows what happened during its 
% execution with an animation.
%
%    INPUT
%     - filename  : name of the gif
%     - mdp       : the MDP to be animated
%     - policy    : the policy to be executed
%     - steps     : max steps of the episode
%     - pausetime : time between animation frames
%     - render    : (optional) to render pixels generated from the MDP

close all

ds = collect_samples(mdp, 1, steps, policy);

if nargin < 6, render = 0; end
if nargin < 5 || isempty(pausetime), pausetime = 0.1; end

F = struct('cdata',[],'colormap',[]);

states = [ds.s(:,1) ds.nexts];

if render
    [pixels, clims, cmap] = mdp.render(states);
end

for i = 1 : size(states,2)
    
    if render
        if i == 1, fig = figure(); fig.Name = 'Pixels Animation'; end
        figure(fig)
        clf, imagesc(pixels(:,:,i))
    else
        if i == 1, mdp.closeplot, mdp.initplot(); end
        mdp.updateplot(states(:,i));
    end
    
    ax = gca;
    ax.Units = 'pixels';
    if render, ax.CLim = clims; colormap(cmap); end
    pos = ax.Position;
    marg = 30;
    rect = [-marg, -marg, pos(3)+2*marg, pos(4)+2*marg];
    F(i) = getframe(ax, rect);

    [X, map] = rgb2ind(frame2im(F(i)),256);
    
    % Write gif
    if i == 1
        imwrite(X, map, [filename '.gif'], 'Loopcount', inf, 'DelayTime', pausetime)
    else
        imwrite(X, map, [filename '.gif'], 'WriteMode', 'Append', 'DelayTime', pausetime)
    end
    
    size(F(i).cdata)
end

% Write video
v = VideoWriter(filename);
v.FrameRate = 1/pausetime;
open(v)
writeVideo(v,F)
close(v)