function [x, t] = rmsprop2(f, fd, x, varargin)
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

hyperparameters = {'alpha', 'rmspropAlpha', 'momentum', 'decay', 'maxiter', 'tolstep', 'tolfun'};
defaults = {1e-3, 0.1, 0.7, 0, 1e5, 1e-6, 1e-6};
% rmspropAlpha : 0 -> deactivated ; 1 -> rprop ; 0.1 -> standard RMSprop

[alpha, rmspropAlpha, momentum, decay, maxiter, tolstep, tolfun] = ...
    internal.stats.parseArgs(hyperparameters, defaults, varargin{:});

if ~iscolumn(x), x = x'; end

m = 1;
v = 0;
t = 0;
converged = false;

while ~converged && t < maxiter
    
    x_old = x;
    fd_eval = fd(x);
    if ~iscolumn(fd_eval), fd_eval = fd_eval'; end
    t = t + 1;
    
    x = x + v - alpha * decay * x; % Nesterov momentum
    m = rmspropAlpha * (fd_eval.^2) + (1 - rmspropAlpha) * m;
    xN = fd_eval ./ sqrt(m + 1e-12);
    v = momentum * v - alpha * xN;
    x = x + v - alpha * decay * x;
    
    converged = norm(x - x_old) < tolstep || norm(f(x) - f(x_old)) < tolfun;
    
end
