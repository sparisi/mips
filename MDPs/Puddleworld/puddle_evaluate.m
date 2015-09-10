function J = puddle_evaluate(policy, doPlot)

J = 0;
c = 0;
n = 30;
policy = policy.makeDeterministic;
x = linspace(0,1,n);
y = linspace(0,1,n);
figure, hold all
for i = 1 : n
    for j = 1 : n
        state = [x(i); y(j)];
        if ~(x(i) > 0.95 && y(j) > 0.95)
            action = policy.drawAction(state);
            [~, r] = puddlec_simulator(state,action);
            J = J + r;
            c = c + 1;
            if nargin > 1
                switch action
                    case 1, text(x(i),y(j),'\leftarrow')
                    case 2, text(x(i),y(j),'\rightarrow')
                    case 3, text(x(i),y(j),'\uparrow')
                    case 4, text(x(i),y(j),'\downarrow')
                end
            end
        end
    end
end
J = J / c;


end