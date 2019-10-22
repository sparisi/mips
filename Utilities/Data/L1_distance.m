function d = L1_distance(a, b)
% L1_DISTANCE - computes Manhattan distance matrix
%
% M = L1_distance(A,B)
%
%    A - (DxM) matrix 
%    B - (DxN) matrix
% 
% Returns:
%    E - (MxN) Manhattan distances between vectors in A and B

    if nargin < 2
       error('Not enough input arguments');
    end
    if size(a, 1) ~= size(b, 1)
        error('A and B should be of same dimensionality');
    end
    if ~isreal(a) || ~isreal(b)
        warning('Computing distance table using imaginary inputs. Results may be off.'); 
    end

    % Padd zeros if necessray
    if size(a, 1) == 1
        a = [a; zeros(1, size(a, 2))]; 
        b = [b; zeros(1, size(b, 2))]; 
    end

    % Compute distance table
    d = sum( abs( bsxfun(@minus, permute(a, [2 3 1]), permute(b, [3 2 1]))), 3);

    % Make sure result is real
    d = real(d);

