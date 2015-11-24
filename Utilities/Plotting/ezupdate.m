function ezupdate(ezHandle, funHandle)
% EZUPDATE Updates a figure plotted with easy-to-use functions (ezplot,
% ezcontourf, ...).
% It does not support ezplot3.
%
%    INPUT
%     - ezHandle  : handle of the graphics element
%     - funHandle : function used for plotting
%
% =========================================================================
% EXAMPLE
% f = @(x,y)x+y.^2;
% h = ezmeshc(f);
% f = @(x,y)x.^3-y;
% pause(0.5)
% ezupdate(h,f)

for i = 1 : numel(ezHandle) % ezmeshc and ezsurfc return two graphics elements
    
    element = ezHandle(i);
    
    X = element.XData;
    Y = element.YData;
    
    if isempty(element)
    elseif isa(element,'matlab.graphics.chart.primitive.Line') % ezplot ezpolar
        element.YData = funHandle(X);
    elseif isa(element,'matlab.graphics.chart.primitive.Contour') ... % ezcontour
            || isa(element,'matlab.graphics.chart.primitive.Surface') % ezsurf ezmesh
        step = size(X,1);
        element.ZData = reshape(funHandle(X(:),Y(:)),step,step);
    else
        error('Unknown easy-to-use element.');
    end
    
end
