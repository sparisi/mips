p1 = [0.1 0.75; % Centers of the first puddle
    0.45 0.75];
p2 = [0.45 0.4; % Centers of the second puddle
    0.45 0.8];

radius = 0.1;

% figure, hold all
puddle1 = [];
puddle2 = [];

for x = 0 : 0.01 : 1
    for y = 0 : 0.01 : 1
        
        state = [x; y];
        
        if state(1) > p1(2,1)
            d1 = norm(state' - p1(2,:));
        elseif state(1) < p1(1,1)
            d1 = norm(state' - p1(1,:));
        else
            d1 = abs(state(2) - p1(1,2));
        end
        
        if state(2) > p2(2,2)
            d2 = norm(state' - p2(2,:));
        elseif state(2) < p2(1,2)
            d2 = norm(state' - p2(1,:));
        else
            d2 = abs(state(1) - p2(1,1));
        end
        
        if d1 > 0 && d1 <= radius
            puddle1 = [puddle1; state'];
        elseif d2 > 0 && d2 <= radius
            puddle2 = [puddle2; state'];
        end
        
    end
end

%%
z1 = zeros(size(puddle1,1),1);
z2 = zeros(size(puddle2,1),1);

hold all

tri = delaunay(puddle1(:,1),puddle1(:,2));
trisurf(tri, puddle1(:,1), puddle1(:,2), z1);
tri = delaunay(puddle2(:,1),puddle2(:,2));
trisurf(tri, puddle2(:,1), puddle2(:,2), z2);

alpha(0.7)
light('Position',[-50 -15 29]);
lighting phong
shading interp
    
rectangle('Position',[0.95,0.95,2,2],'FaceColor',[1 .5 .5],'EdgeColor','r',...
    'LineWidth',3)

axis([0 1 0 1])
