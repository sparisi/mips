function R = nds(P)
% Non-domination sorting. Duplicate solutions (same rows in P) are ignored
% for the dominance count and are assigned all the same rank.
%
% Inputs:
% - P : N-by-D matrix, where N is the number of points and D is the
%       number of elements (objectives) of each point.
%
% Outputs:
% - R : N-by-3 matrix. First column has the number of solutions by which
%       each solution is weakly dominated. Second column has the sub-front
%       where each solution belongs. Third column has a crowding
%       distance.
%
% NB: The crowding distance used in this code is not exactly the one
% described in NSGA-II. In this implementation, it is the average Euclidean 
% distance between each solution and the solutions belonging to the same 
% sub-front.
% Also notice that the distance is negated to sort the solution correctly 
% (as we prefer sparse solution, the higher the distance the better).
%
% Reference: K Deb, A Pratap, S Agarwal, T Meyarivan,
% "A fast and elitist multiobjective genetic algorithm: NSGA-II",
% IEEE Transactions on Evolutionary Computation, 2002.
%
% Example:
%
% P = [0.6850   0.2048
%     0.2649    0.2967
%     0.3348    0.6344
%     0.3037    0.4429
%     0.3659    0.0754
%     0.0220    0.5690
%     0.5135    0.3447
%     0.1898    0.4758]
%
% nds(P)
%     0    0    -0.3878
%     3    2    -Inf
%     0    0    -0.4473
%     1    1    -0.2666
%     2    1    -0.4706
%     1    1    -0.3674
%     0    0    -0.2809
%     1    1    -0.2493
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
%     0     0     -0.4473   |   0.6850    0.2048
%     0     0     -0.3878   |   0.3348    0.6344  <- Pareto-front solutions
%     0     0     -0.2809   |   0.5135    0.3447
%     -----------------------------------------
%     2     1     -0.4706   |   0.3659    0.0754
%     1     1     -0.3674   |   0.3037    0.4429
%     1     1     -0.2666   |   0.1898    0.4758  <- First subfront
%     1     1     -0.2493   |   0.0220    0.5690
%     -----------------------------------------
%     3     2     -Inf      |   0.2649    0.2967  <- Second subfront

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
    subfront = bsxfun(@times,1./max(subfront),subfront); % Normalize the objectives
    C = mat2cell(subfront,ones(size(subfront,1),1),d);
    subdist = cellfun( ...
        @(X)mean( nonzeros( sqrt( sum( bsxfun(@plus, X, -subfront).^2, 2) ) ) ), ...
        C, 'UniformOutput', false );
    % We take only 'nonzeros' elements because 0 distances are due to duplicates
    avgdist(idx_P) = vertcat(subdist{:});
    
end

avgdist(isnan(avgdist)) = Inf; % NaN will occur if a subfront has only one element

% Count the number of solutions by which every solution is weakly dominated
C = mat2cell(P,ones(n,1),d);
dominance = cellfun( ...
    @(X) sum( sum( bsxfun(@ge, uniqueP, X), 2) == d ), ...
    C, 'UniformOutput', false);
ndominate = vertcat(dominance{:}) - 1;

% Since we prefer sparse solutions, we negate the distance
R = [ndominate, rank, -avgdist];
