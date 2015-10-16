function fig = puddle2_plotdet(policy, fig)
% Plots the behavior of the policy and a map of the immediate reward.

domain = 'puddle2';
n = 30;
policy = policy.makeDeterministic;
x = linspace(0,1,n);
y = linspace(0,1,n);
[X, Y] = meshgrid(x,y);
states = [X(:)'; Y(:)'];
actions = policy.drawAction(states);
ntot = size(states,2);
nexts = zeros(size(states));
r = zeros(1,ntot);

parfor i = 1 : ntot
    nexts(:,i) = feval([domain '_simulator'], states(:,i), actions(:,i));
    [~, r(i)] = feval([domain '_simulator'], states(:,i), [0;0]);
end

if nargin > 1, figure(fig); else fig = figure; end
contourf(X, Y, reshape(r,n,n)), colormap(hot), colorbar
hold on
d = nexts-states;
quiver(states(1,:),states(2,:),d(1,:),d(2,:))
axis([0 1 0 1])

end
