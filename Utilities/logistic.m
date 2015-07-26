function val = logistic(weights, tau)
    tau = max(tau,1e-4); % tau must be positive
    min_val = 1e-4; % to avoid numerical problems
	val = tau ./ ( ones(length(weights),1) + exp(-weights) );
    val = max(min_val, val);
return