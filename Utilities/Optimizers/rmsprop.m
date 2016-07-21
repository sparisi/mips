function [x, t] = rmsprop(f, fd, x, varargin)
% RMSPROP Stochastic gradient descent optimizer.
%
%    INPUT
%     - f  : anonymous function, function to be minimized
%     - fd : anonymous function, derivative of the function to be minimized
%     - x  : initial parameters vector
%
%    OUTPUT
%     - x  : optimal parameters (column vector)
%     - t  : total timesteps needed
%
% =========================================================================
% REFERENCE
% T Tieleman, G Hinton
% Lecture 6.5 - COURSERA: Neural Networks for Machine Learning (2012)

hyperparameters = {'alpha', 'gamma', 'beta', 'epsilon', 'maxiter', 'tolstep', 'tolfun'};
defaults = {1e-3, 0.9, 0.9, 1e-8, 1e5, 1e-6, 1e-6};

[alpha, gamma, beta, epsilon, maxiter, tolstep, tolfun] = ...
    internal.stats.parseArgs(hyperparameters, defaults, varargin{:});

if ~iscolumn(x), x = x'; end

m = 1;
v = 0;
t = 0;
converged = false;

while ~converged && t < maxiter
    
    x_old = x;
    t = t + 1;
    x = x - beta * v; % Nesterov momentum
    fd_eval = fd(x);
    if ~iscolumn(fd_eval), fd_eval = fd_eval'; end
    
    m = gamma * m + (1 - gamma) * fd_eval.^2;
    v = beta * v + alpha ./ sqrt(m + epsilon) .* fd_eval;
    x = x - v;

    converged = norm(x - x_old) < tolstep || norm(f(x) - f(x_old)) < tolfun;
    
end
