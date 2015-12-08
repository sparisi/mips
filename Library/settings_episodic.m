function [n_obj, policy] = settings_episodic(domain, isDet)
% SETTING_EPISODIC Wrapper to retrieve the settings for episodic algorithms.
%
%    INPUT
%     - domain : name of the MDP
%     - isDet  : 1 if you want the low-level policy to be deterministic, 
%                0 otherwise
%
%    OUTPUT
%     - n_obj  : number of the objective of the MDP
%     - policy : high-level policy, i.e., distribution used to draw the
%                parameters of the low-level

[n_obj, pol_low] = feval([domain '_settings']);
n_params = length(pol_low.theta) - pol_low.dim_explore * isDet;
mu0 = pol_low.theta(1:n_params);

%%% These are suggested generic covariances 
% sigma0 = 10 * eye(n_params); % Deep
sigma0 = 100 * eye(n_params); % Dam
% sigma0 = 0.1 * eye(n_params); % LQR

sigma0 = sigma0 + diag(abs(mu0));
sigma0 = nearestSPD(sigma0);

% policy = gaussian_constant(n_params,mu0,sigma0); % REPS
policy = gaussian_chol_constant(n_params,mu0,sigma0); % REPS / NES
% policy = gaussian_diag_constant(n_params,mu0,sigma0); % NES / REPS
% policy = gmm_constant(mu0,sigma0,5); % REPS

end
