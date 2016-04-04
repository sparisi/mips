function S = metric_nd(J)
% METRIC_ND Scalarizes a set of solution for multi-objective problem (i.e., 
% an approximate Pareto frontier). A solution is ranked according to the
% numer of other solutions dominating it. For more details, see NDS2.
%
%    INPUT
%     - J : [N x M] matrix of samples to evaluate, where N is the number of
%           samples and M the number of objectives
%
%    OUTPUT
%     - S : [N x 1] vector of the non-dominance-based metric value for each
%           sample

C = nds2(J);
S = -(C(:,1) + C(:,3));