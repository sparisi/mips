function [J, ds] = show_simulation(mdp, policy, pausetime, steps, render)
% SHOW_SIMULATION Runs an episode and shows what happened during its 
% execution with an animation.
%
%    INPUT
%     - mdp       : the MDP to be seen
%     - policy    : the low level policy
%     - steps     : steps of the episode
%     - pausetime : time between animation frames
%     . render    : (optional) to render pixels generated from the MDP
%
%    OUTPUT
%     - J         : the total return of the episode
%     - ds        : the episode dataset

mdp.closeplot
[ds, J] = collect_samples(mdp, 1, steps, policy);

if nargin < 5, render = 0; end

if ~render
    mdp.plotepisode(ds, pausetime)
else
    pixels = mdp.render([ds.s(:,1), ds.nexts]);
    fig = findobj('type','figure','name','Pixels Animation');
    if isempty(fig), fig = figure(); fig.Name = 'Pixels Animation'; end
    try colormap(mdp.cmap); catch, end
    for i = 1 : size(pixels,3)
        clf, image(pixels(:,:,i),'CDataMapping','scaled'), drawnow limitrate, pause(pausetime)
    end
end
