function dam_animation(episode)

env = dam_environment;
S = env.S; % surface
x = sqrt(S); y = x; % assume square surface

vert = @(z1,z2)[x y z1; % cube vertices
    0 y z1;
    0 y z2;
    x y z2;
    0 0 z2;
    x 0 z2;
    x 0 z1;
    0 0 z1];

fac = [1 2 3 4; % cube faces
    4 3 5 6;
    6 7 8 5;
    1 2 8 7;
    6 7 1 4;
    2 3 5 8];

figure
mass = episode.s;
for k = 1 : length(mass)
    clf, hold all
    z = mass(k) / S;
    patch('Faces', fac, 'Vertices', vert(0,z), 'FaceColor', 'b'); % Current mass
    patch('Faces', fac, 'Vertices', vert(env.H_FLO_U-1,env.H_FLO_U+1), 'FaceColor', 'r'); % Flooding threshold
    
    a = episode.a(k);
    amin = max(z - env.S_MIN_REL, 0);
    amax = z;    
    if a < amin || a > amax, c = 'r'; else c = 'g'; end % Check if action is valid
    title(num2str(episode.a(k)), 'color', c) % If not valid display the action in red

    axis([0, 1, 0, 1, 0, 200])
    view(3)
    drawnow
    pause(0.1)
end

end
