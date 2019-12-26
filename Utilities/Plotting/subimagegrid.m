function subimagegrid(name, X, Y, Z, addcolorbar)
% Like SUBIMAGESC, but axis are hidden and a grid is added. 

[nplots, ~] = size(Z);
m = length(X);
n = length(Y);
nrows = floor(sqrt(nplots));
ncols = ceil(nplots/nrows);

f = findobj('type','figure','name',name);

if isempty(f)
    if nargin == 4, addcolorbar = 0; end
    figure('DefaultAxesPosition', [0.1, 0.1, 0.8, 0.8], 'Name', name);
    for i = nplots : -1 : 1
        subplot(nrows,ncols,i,'align')
        h = image(reshape(Z(i,:),n,m));
        imggrid(h,'k',0.5); % Add grid
        if nplots > 1, title(num2str(i)), end
        caxis manual
        caxis([min(Z(:)) max(Z(:))]);
        if addcolorbar, colorbar, end
    end
else
    axes = findobj(f,'type','axes');
    for i = nplots : -1 : 1
        axes(i).Children(3).CData = reshape(Z(i,:),n,m);
        axes(i).CLim = [min(Z(:)) max(Z(:))];
    end
end

drawnow
