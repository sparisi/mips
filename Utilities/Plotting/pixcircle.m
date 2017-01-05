function c = pixcircle(w, h, cx, cy, r)
% PIXCIRCLE Generates a matrix of [0,1], where the 1 denotes the area
% covered by a circle. 
%
%    INPUT
%     - w  : width of the image (in pixels)
%     - h  : height of the image (in pixels)
%     - cx : x center of the circle (pixel coord)
%     - cy : y center of the circle (pixel coord)
%     - r  : radius of the circle (in pixels)
% 
%    OUTPUT
%     - c  : [W x H] matrix 

[x,y] = meshgrid(-(cx-1):(w-cx), -(cy-1):(h-cy));
c = ((x.^2+y.^2) <= r^2);