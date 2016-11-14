function R = nds2(P)
% NDS2 Non-domination sorting. Duplicate solutions (same rows in P) are 
% ignored for the dominance count and are assigned all the same rank.
% The crowding distance is the average normalized Euclidean distance 
% between each solution and the solutions belonging to the same sub-front. 
% For the original sorting, please see NDS.
%
%    INPUT
%     - P : [N x D] matrix, where N is the number of points and D is the
%           number of elements (objectives) of each point.
%
%    OUTPUT
%     - R : [N x 3] matrix. First column has the number of solutions by 
%           which each solution is weakly dominated. Second column has the 
%           sub-front where each solution belongs. Third column has a 
%           crowding distance.
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
%     0.1898    0.4758]
%
% R = nds2(P)
%     0    0    -0.3515
%     3    2    -1.0000
%     0    0    -0.3967
%     1    1    -0.2019
%     2    1    -0.3231
%     1    1    -0.2877
%     0    0    -0.2517
%     1    1    -0.1873
%
% We can then sort R and use the indices to sort P.
% [Rs, idx] = sortrows(R);
% Ps = P(idx,:);
% 
% Here is an analysis of [Rs, Ps] components and how solutions are sorted 
% according to the algorithm:
%
%              Rs                     Ps
%     #dom  subf  avg dist  |    Obj1      Obj2
%     -----------------------------------------
%     0     0     -0.3967   |   0.3348    0.6344
%     0     0     -0.3515   |   0.6850    0.2048  <- Pareto-front solutions
%     0     0     -0.2517   |   0.5135    0.3447
%     -----------------------------------------
%     1     1     -0.2877   |   0.0220    0.5690
%     1     1     -0.2019   |   0.3037    0.4429
%     1     1     -0.1873   |   0.1898    0.4758  <- First subfront
%     2     1     -0.3231   |   0.3659    0.0754
%     -----------------------------------------
%     3     2      1.0000   |   0.2649    0.2967  <- Second subfront

tmp = P;
uniqueP = unique(P,'rows');
[n, d] = size(P);
rank = zeros(n,1);
avgdist = zeros(n,1);
j = 0;

% Extract all the sub-fronts and rank each solution. First, Pareto-front 
% solutions are identified, given rank 0 and deleted. Then, iteratively, 
% sub-fronts are identified and their rank is progressively increased.
while ~isempty(tmp)
    
    [subfront, ~, idx_tmp] = pareto(tmp);
    
    % Indices (because of possible duplicates)
    idx_P = ismember(P, subfront, 'rows');
    
    % Assign the rank
    rank(idx_P) = j;
    tmp(idx_tmp,:) = [];
    j = j + 1;

    % Assign the crowding distance
    subfront = P(idx_P,:); % Include duplicates for correct indexing
    subfront = normalize_data(subfront,min(subfront),max(subfront)); % Normalize the objectives
    C = mat2cell(subfront,ones(size(subfront,1),1),d);
    subdist = cellfun( ...
        @(X)mean( nonzeros( sqrt( sum( bsxfun(@plus, X, -subfront).^2, 2) ) ) ), ...
        C, 'UniformOutput', false );
    
    % We take only 'nonzeros' elements because 0 distances are due to duplicates
    avgdist(idx_P) = vertcat(subdist{:});
    avgdist(idx_P) = avgdist(idx_P) / sum(avgdist(idx_P)); % Normalize distance

end

avgdist(isnan(avgdist)) = 1; % NaN will occur if a subfront has only one element

% Count the number of solutions by which every solution is weakly dominated
C = mat2cell(P,ones(n,1),d);
dominance = cellfun( ...
    @(X) sum( sum( bsxfun(@ge, uniqueP, X), 2) == d ), ...
    C, 'UniformOutput', false);
ndominate = vertcat(dominance{:}) - 1;

% Since we prefer sparse solutions, we negate the distance
R = [ndominate, rank, -avgdist];