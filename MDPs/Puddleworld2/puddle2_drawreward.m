function h = puddle2_drawreward()

domain = 'puddle2';
n = 30;
x = linspace(0,1,n);
y = linspace(0,1,n);
allstates = [];
allr = [];
parfor i = 1 : n
    for j = 1 : n
        state = [x(i); y(j)];
        action = [0; 0];
        allstates = [allstates; state'];
        [~, r] = feval([domain '_simulator'],state,action);
        allr = [allr, r];
    end
end

h = surfFromScatter(allstates(:,1), allstates(:,2), allr, 0.5);

end