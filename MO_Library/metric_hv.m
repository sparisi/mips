function S = metric_hv(J, HVF)
% METRIC_HV Scalarizes a set of solution for multi-objective problem (i.e., 
% an approximate Pareto frontier). A solution is ranked according to its
% contribution to the hypervolume of the frontier.
%
%    INPUT
%     - J   : N-by-M matrix of samples to evaluate, where N is the number
%             of samples and M the number of objectives
%     - HVF : hypervolume function handle
%
%    OUTPUT
%     - S   : N-by-1 vector of the hypervolume contribution of each sample

[uniqueJ, ~, idx] = unique(J,'rows'); % avoid duplicates to save time
front = pareto(uniqueJ);
hyperv_ref = HVF(front);

idx2 = find(all(ismember(uniqueJ,front),2)); % dominated solutions metric value is 0
hvUnique = zeros(size(uniqueJ,1),1);
hvContrib = zeros(size(front,1),1);
parfor i = 1 : size(front,1)
    front_tmp = front;
    front_tmp(i,:) = [];
    hvContrib(i) = hyperv_ref - HVF(pareto(front_tmp));
end
hvUnique(idx2) = hvContrib;
S = hvUnique(idx); % map back to duplicates