function hv = hypervolume(F, AU, U, N)
% Approximates the hypervolume of a Pareto frontier. First, it generates 
% random samples in the hypercuboid defined by the utopia and antiutopia 
% points. Second, it counts the number of samples dominated by the front. 
% The hypervolume is approximated as the ratio 'dominated points / total
% points'.
% Please notice that the choice of the utopia and antiutopia point is
% crucial: using points very far from the frontier will result in similar
% hypervolume even for very different frontiers (if the utopia is too far 
% away, the hypervolume will be always low; if the antiutopia is too far 
% away, the hypervolume will be always high). 
% Also, frontier points "beyond" the reference points will not be counted 
% for the approximation (e.g., if the antiutopia is above the frontier or 
% the utopia is below, the hypervolume will be 0).
%
% Inputs:
% - F  : the Pareto front to evaluate
% - AU : antiutopia point
% - U  : utopia point
% - N  : number of sample for the approximation
%
% Outputs:
% - hv : hypervolume

[n_sol, dim] = size(F);

samples = bsxfun( @plus, AU, bsxfun(@times, (U - AU), rand(N, dim)) );

dominated = 0;
for i = 1 : n_sol
    idx = sum( bsxfun( @ge, F(i,:), samples ), 2 ) == dim;
    dominated = dominated + sum(idx);
    samples(idx,:) = [];
end

hv = dominated / N;

return
