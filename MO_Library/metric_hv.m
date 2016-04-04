function S = metric_hv(J, HVF)
% METRIC_HV Scalarizes a set of solution for multi-objective problem (i.e., 
% an approximate Pareto frontier). A solution is ranked according to its
% contribution to the hypervolume of the frontier.
%
%    INPUT
%     - J   : [N x M] matrix of samples to evaluate, where N is the number
%             of samples and M the number of objectives
%     - HVF : hypervolume function handle
%
%    OUTPUT
%     - S   : [N x 1] vector of the hypervolume contribution of each sample
%
% =========================================================================
% REFERENCE
% N Beume, B Naujoks, M Emmerich
% SMS-EMOA: Multiobjective selection based on dominated hypervolume (2007)

[uniqueJ, ~, idx] = unique(J,'rows'); % Ignore duplicates
[front, ~, idx2] = pareto(uniqueJ); % Dominated solutions have 0 contribution

hyperv_ref = HVF(front);

parfor i = 1 : size(front,1)
    front_tmp = front;
    front_tmp(i,:) = [];
    hvContrib(i) = hyperv_ref - HVF(pareto(front_tmp));
end

hvUnique = zeros(size(uniqueJ,1),1);
hvUnique(idx2) = hvContrib;
S = hvUnique(idx); % Map back to duplicates
S = max(0,S); % The contribution estimation might be negative using a Monte Carlo approx of the hypervolume
S(S==0) = -0.1; % Penalty for dominated solutions
