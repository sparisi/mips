function r = argmax(x, dim)

if nargin == 1
    if isvector(x)
        [~, r] = max(x);
        return
    else
        error('Not enough input arguments.')
    end
end

[~, r] = max(x, [], dim);
