function c = pixrectangle(w, h, cx, cy, xr, yr)
% PIXRECTANGLE Generates a matrix of [0,1], where the 1 denotes the area
% covered by a rectangle. 
%
%    INPUT
%     - w  : width of the image (in pixels)
%     - h  : height of the image (in pixels)
%     - cx : x center of the rectangle (pixel coord)
%     - cy : y center of the rectangle (pixel coord)
%     - xr : x radius of the rectangle (in pixels)
%     - yr : y radius of the rectangle (in pixels)
% 
%    OUTPUT
%     - c  : [W x H] matrix 

[x,y] = meshgrid(-(cx-1):(w-cx), -(cy-1):(h-cy));
c = (abs(x)<=xr) & (abs(y) <= yr);
