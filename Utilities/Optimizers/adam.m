function [x, t] = adam(f, fd, x, varargin)
% ADAM Stochastic gradient descent optimizer. 
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
% D P Kingma, J L Ba
% Adam: A Method for Stochastic Optimization

hyperparameters = {'alpha', 'beta1', 'beta2', 'epsilon', 'lambda', 'maxiter', 'tolstep', 'tolfun'};
defaults = {1e-3, 0.9, 0.999, 1e-8, 0, 1e5, 1e-6, 1e-6};

[alpha, beta1, beta2, epsilon, lambda, maxiter, tolstep, tolfun] = ...
    internal.stats.parseArgs(hyperparameters, defaults, varargin{:});

if ~iscolumn(x), x = x'; end

assert(beta1 < 1 && beta2 < 1 && beta1 >= 0 && beta2 >= 0, ...
    'Hyperparameters beta must be in [0,1).')

m = 0; % Init 1st moment vector
v = 0; % Init 2nd moment vector
t = 0;
converged = false;

while ~converged && t < maxiter

    x_old = x;
    t = t + 1;
    fd_eval = fd(x); % Evaluate gradient wrt x at timestep t
    if ~iscolumn(fd_eval), fd_eval = fd_eval'; end
    
    m = beta1 * m + (1 - beta1) * fd_eval; % Update biased 1st moment estimate
    v = beta2 * v + (1 - beta2) * fd_eval.^2; % Update biased 2nd raw moment estimate
    mhat = m / (1 - beta1^t); % Compute bias-corrected 1st moment estimate
    vhat = v / (1 - beta2^t); % Compute bias-corrected 2nd raw moment estimate
    x = x - alpha * mhat ./ (sqrt(vhat) + epsilon) ...
        - alpha * lambda * x; % Update with l2 weight decay

    converged = norm(x - x_old) < tolstep || norm(f(x) - f(x_old)) < tolfun;
    
end
