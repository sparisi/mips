function [n_obj, policy] = settings_episodic(domain, isDet)
% Function used to retrieve the settings for episodic algorithms.
%
% Inputs: 
% - domain : name of the MDP
% - isDet  : 1 if you want the low-level policy to be deterministic, 
%            0 otherwise
%
% Outputs:
% - n_obj  : number of parameters of the low-level policy to be learnt
% - policy : high-level policy, i.e., distribution used to draw low-level 
%            policy parameters

[n_obj, pol_low] = feval([domain '_settings']);
n_params = length(pol_low.theta) - pol_low.dim_explore * isDet;
mu0 = pol_low.theta(1:n_params);

%%% These are suggested generic covariances 
sigma0 = 10 * eye(n_params); % Deep
% sigma0 = 100 * eye(n_params); % Dam
% sigma0 = 0.1 * eye(n_params); % LQR

sigma0 = sigma0 + diag(abs(mu0));

%%% Single Gaussian
policy = gaussian_constant(n_params,mu0,sigma0); % REPS
% policy = gaussian_chol_constant(n_params,mu0,chol(sigma0)); % REPS
% policy = gaussian_diag_constant(n_params,mu0,sqrt(diag(sigma0))); % NES / REPS

%%% Mixture Model
% n_gauss = 5;
% mu = zeros(n_gauss,n_params);
% sigma = zeros(n_params,n_params,n_gauss);
% for i = 1 : n_gauss
%     mu(i,:) = policy.drawAction;
%     sigma(:,:,i) = sigma0;
% end
% p = ones(n_gauss,1) / n_gauss;
% policy = gmm_constant(mu,sigma,p,n_gauss);

end
