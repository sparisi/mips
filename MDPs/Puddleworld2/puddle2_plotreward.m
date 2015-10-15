function puddle2_plotreward()

domain = 'puddle2';
n = 50;
x = linspace(0,1,n);
y = linspace(0,1,n);
Z = [];
parfor i = 1 : n
    for j = 1 : n
        state = [x(i); y(j)];
        action = [0; 0];
        [~, r] = feval([domain '_simulator'],state,action);
        Z(i,j) = r;
    end
end

[X, Y] = meshgrid(x,y);
figure, contourf(X, Y, Z)
colorbar

end