function f = squash(x)
% SQUASH Generalized logistic squashing function for input X in [0,1].
% The function intercepts the y-axis in M (0.5).

A = 0;
K = 1;
B = 10; % change it to change the steepness
Q = 1;
v = 1;
M = 0.5;

f = A + ( K - A ) ./ ( 1 + Q*exp(-B*(x-M)) ).^(1/v);