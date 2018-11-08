function [f, df, ddf] = mse_linear(w, X, T)
% Mean squared error for linear functions
% min_w ||Y-T||^2,   Y = w'X

Y = w'*X;
E = Y - T; % T are the targets
f = 0.5*mean(E.^2); % Function, MSE
df = X*E'/size(T,2); % Gradient
ddf = X*X'/size(T,2); % Hessian
