function [g, gd, h] = mse_linear(w, X, T)
% Mean squared error for linear functions.

Y = w'*X;
TD = Y - T; % T are the targets (constant)
g = 0.5*mean(TD.^2);
gd = X*TD'/size(T,2);
h = X*X'/size(T,2);
end
