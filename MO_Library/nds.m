function R = nds(P)
% NDS Non-domination sorting. Duplicate solutions (same rows in P) are 
% ignored for the dominance count and are assigned all the same rank.
%
%    INPUT
%     - P : [N x D] matrix, where N is the number of points and D is the
%           number of elements (objectives) of each point.
%
%    OUTPUT
%     - R : [N x 2] matrix. First column has the sub-front where each 
%           solution belongs. Second column has the crowding distance.
%
% =========================================================================
% WARNING
% The crowding distance is negated to sort solutions correctly. 
% Otherwise, a sort on the matrix would order it incorrectly, as the rule 
% for the rank is "the lower, the better", while for the distance is 
% "the higher, the better".
%
% =========================================================================
% REFERENCE
% K Deb, A Pratap, S Agarwal, T Meyarivan
% A fast and elitist multiobjective genetic algorithm: NSGA-II (2002)
%
% =========================================================================
% EXAMPLE
% P = [0.6850   0.2048
%     0.2649    0.2967
%     0.3348    0.6344
%     0.3037    0.4429
%     0.3659    0.0754
%     0.0220    0.5690
%     0.5135    0.3447
%     0.1898    0.4758];
%
% R = nds(P)
%     0      -Inf
%     2      -Inf
%     0      -Inf
%     1   -1.3233
%     1      -Inf
%     1      -Inf
%     0   -2.0000
%     1   -1.0746
%
% [Rs, idx] = sortrows(R);
% Ps = P(idx,:);
% 
%                 Rs              |         Ps
%     -----------------------------------------------
%     rank   crowd. dist.   #dom. |    Obj1      Obj2
%     0          -Inf         0   |  0.6850    0.2048
%     0          -Inf         0   |  0.3348    0.6344
%     0       -2.0000         0   |  0.5135    0.3447
%     -----------------------------------------------
%     1          -Inf         1   |  0.0220    0.5690
%     1          -Inf         2   |  0.3659    0.0754
%     1       -1.3233         1   |  0.3037    0.4429
%     1       -1.0746         1   |  0.1898    0.4758
%     -----------------------------------------------
%     2          -Inf         3   |  0.2649    0.2967
%
% (#dom column has been manually added)

tmp = P;
[n, d] = size(P);
rank = zeros(n,1);
avgdist = zeros(n,1);
j = 0;

% Extract all the sub-fronts and rank each solution. First, Pareto-front 
% solutions are identified, given rank 0 and deleted. Then, iteratively, 
% sub-fronts are identified and their rank is progressively increased.
while ~isempty(tmp)
    
    subfront = pareto(tmp);
    
    % Indices (because of possible duplicates)
    idx_P = ismember(P, subfront, 'rows');
    idx_tmp = ismember(tmp, subfront, 'rows');
    
    % Assign the rank
    rank(idx_P) = j;
    tmp(idx_tmp,:) = [];
    j = j + 1;

    % Assign the crowding distance
    subfront = P(idx_P,:); % Include duplicates for correct indexing
    subfront = normalize_points(subfront,min(subfront),max(subfront)); % Normalize the objectives

    avgdist(idx_P) = crowding_distance(subfront);

end

R = [rank, -avgdist];



function dist = crowding_distance(sub_P)

[n, d] = size(sub_P);
tmp = [sub_P, zeros(n,1)];
for m = 1 : d
    tmp = sortrows(tmp,m);
    tmp([1,n],end) = inf; % Boundary solutions have max distance
    tmp(2:end-1,end) = tmp(2:end-1,end) + ...
        (tmp(3:end,m) - tmp(1:end-2,m)) / (max(tmp(:,m)) - min(tmp(:,m)));
end

% Original ordering
[~, idx] = sortrows(sub_P,d);
[~, idx] = sort(idx);

dist = tmp(idx,end);
