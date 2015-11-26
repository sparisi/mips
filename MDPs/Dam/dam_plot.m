function dam_plot(state, action)

persistent figureHandle agentHandle

env = dam_environment;
S = env.S; % surface

x = sqrt(S); y = x; % assume square surface

vertices = @(z1,z2)[x y z1; % cube vertices
    0 y z1;
    0 y z2;
    x y z2;
    0 0 z2;
    x 0 z2;
    x 0 z1;
    0 0 z1];

faces = [1 2 3 4; % cube faces
    4 3 5 6;
    6 7 8 5;
    1 2 8 7;
    6 7 1 4;
    2 3 5 8];

if isempty(findobj('type','figure','name','Dam Plot'))
    figureHandle = figure();
    figureHandle.Name = 'Dam Plot';
    hold all
    
    patch('Faces', faces, 'Vertices', vertices(env.H_FLO_U-1,env.H_FLO_U+1), 'FaceColor', 'r'); % Flooding threshold
    agentHandle = patch('Faces', faces, 'Vertices', vertices(env.H_FLO_U-1,env.H_FLO_U+1), 'FaceColor', 'b'); % Current mass
    
    axis([0, 1, 0, 1, 0, 200])
    view(3)
end

if nargin == 0
    return
end

z = state / S;
agentHandle.Vertices = vertices(0,z);

title('')
if nargin == 2
    amin = max(z - env.S_MIN_REL, 0);
    amax = z;
    if action < amin || action > amax, c = 'r'; else c = 'g'; end % Check if action is valid
    title(num2str(action), 'color', c) % If not valid display the action in red
end

end
