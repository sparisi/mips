function saveFigAsPDF ( filename, fig )

if nargin == 1, fig = gcf(); end

w = 5;
h = 5;
set(fig.CurrentAxes, 'LooseInset', get(fig.CurrentAxes, 'TightInset'));
set(fig, 'PaperPosition', [0 0 w h]);
set(fig, 'PaperSize', [w h]);
saveas(fig, [filename '.pdf'])
