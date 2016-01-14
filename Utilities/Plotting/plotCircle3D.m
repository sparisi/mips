function h = plotCircle3D(center, normal, radius, color, isfilled)
% Plots a circle in 3d and returns its handle.
% 
% =========================================================================
% ACKNOWLEDGEMENT
% http://www.mathworks.com/matlabcentral/fileexchange/26588-plot-circle-in-3d

theta = 0 : 0.01 : 2*pi;
v = null(normal);
points = repmat(center',1,size(theta,2)) ... 
    + radius * (v(:,1) * cos(theta) ...
    + v(:,2) * sin(theta));

if isfilled
    h = fill3(points(1,:),points(2,:),points(3,:),color);
else
    h = plot3(points(1,:),points(2,:),points(3,:),color);
end

end