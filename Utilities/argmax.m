function r = argmax(x, dim)

[~, r] = max(x, [], dim);
