function f = logistic(x, tau)
% LOGISTIC Logistic function: f = tau / ( 1 + exp(-x) ) with tau > 0

assert(isvector(x), 'Input x must be a vector.')
if isrow(x), x = x'; end

tau = max(tau,1e-8); % tau must be positive
min_val = 1e-8; % to avoid numerical problems
f = tau ./ ( ones(length(x),1) + exp(-x) );
f = max(min_val, f);
