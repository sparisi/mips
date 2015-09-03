function [x, y, z] = extractPoints(h)
% EXTRACTPOINTS Returns X, Y, Z coordinates of a plot given its handle H.

if nargin == 0
    h = gcf; % current figure handle
end
axesObjs = get(h, 'Children'); % axes handles
dataObjs = get(axesObjs, 'Children'); % handles to low-level graphics objects in axes
x = get(dataObjs, 'XData')'; % data from low-level grahics objects
y = get(dataObjs, 'YData')';
z = get(dataObjs, 'ZData')';