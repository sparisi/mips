function avgJ = puddle2_evaluate(policy, h)

domain = 'puddle2';
avgJ = 0;
c = 0;
n = 30;
policy = policy.makeDeterministic;
x = linspace(0,1,n);
y = linspace(0,1,n);
allstates = [];
allnexts = [];
parfor i = 1 : n
    for j = 1 : n
        if ~(x(i) > 0.95 && y(j) > 0.95)
            state = [x(i); y(j)];
            action = policy.drawAction(state);
            allstates = [allstates; state'];
            [next, r] = feval([domain '_simulator'],state,action);
            allnexts = [allnexts; next'];
            avgJ = avgJ + r;
            c = c + 1;
        end
    end
end
avgJ = avgJ / c;

if nargin > 1
    figure(h)
    d = allnexts-allstates;
    quiver(allstates(:,1),allstates(:,2),d(:,1),d(:,2))
    axis([0 1 0 1])
end

end
