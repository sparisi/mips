function f = logistic(x, tau)
% LOGISTIC Logistic function: f = tau / ( 1 + exp(-x) ) with tau > 0

assert(isvector(x), 'Input x must be a vector.')

tau = max(tau,1e-4); % tau must be positive
min_val = 1e-4; % to avoid numerical problems
f = tau ./ ( ones(length(x),1) + exp(-x) );
f = max(min_val, f);
