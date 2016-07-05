% It prints the agent behavior defined by a policy on the environment, a
% contour with the actions drawn by the agent and the basis functions used
% by the policy.

close all

s = multigrid(40,[mdp.stateLB(1) mdp.stateUB(1)],[mdp.stateLB(2) mdp.stateUB(2)]);
a = policy.makeDeterministic.drawAction(s);

if ismember('PolicyDiscrete',superclasses(policy)) % Puddleworld
    steps = [-mdp.step mdp.step 0 0; 0 0 mdp.step -mdp.step];
    a = steps(:,a);
else % PuddleworldContinuous
    a = bsxfun(@times,a,1./matrixnorms(a,2))*mdp.step;
end

%% Plot policy on the environment
mdp.showplot
hold on
ntot = size(s,2);
quiver(s(1,:),s(2,:),a(1,:),a(2,:),'color','r')
axis([mdp.stateLB(1) mdp.stateUB(1) mdp.stateLB(2) mdp.stateUB(2)])
title('Learned Policy')

%% Plot x-action contour
n = 30;
figure
pointsToSurf(s(1,:), s(2,:), a(1,:), n, n, 1)
view(0,90)
colorbar
title('Action on the x-axis')

%% Plot y-action contour
figure
pointsToSurf(s(1,:), s(2,:), a(2,:), n, n, 1)
view(0,90)
colorbar
title('Action on the y-axis')

%% Plot rbf
multisurf(@(x)(policy.basis(x)),mdp.stateLB(1),mdp.stateUB(1),mdp.stateLB(2),mdp.stateUB(2),100)
view(35,30)
title('Basis Functions')

autolayout
