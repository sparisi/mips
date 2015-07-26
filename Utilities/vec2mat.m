function mat = vec2mat(vec, n_rows)

n_col = (length(vec)) / n_rows;

mat = (reshape(vec, n_rows, n_col));

return