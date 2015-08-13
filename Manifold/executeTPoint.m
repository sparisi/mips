function [Jr, D_r_Jr] = executeTPoint(domain, t_point, theta_iter, D_t_theta, ...
    D_t_theta_iter, D_rho_theta_iter, t, rho, rho_learned, ind_type, beta)

[ n_obj, pol, episodes, steps, gamma ] = feval([domain '_settings']);

% Get dimensions
dim_t = length(t);
dim_rho = length(rho);

% Get closed-form derivatives
theta_i = double(subs(theta_iter, t, t_point));
D_t_theta_i = double(subs(D_t_theta_iter, t, t_point));
D_rho_theta_i = double(subs(D_rho_theta_iter, t, t_point));

% Collect samples and estimate gradients and hessians (ReLe)
pol.theta = theta_i;
[~, J, G, H] = collect_samples_rele(domain, episodes, steps, pol);

% % Collect samples and estimate gradients and hessians (Matlab)
% [ds, J] = collect_samples(domain, episodes, steps, pol);
% G2 = zeros(length(theta_i), n_obj);
% for i = 1 : n_obj
%     G2(:,i) = GPOMDPbase(pol, ds, gamma, i);
%     H2{i,1} = HessianRFbase(pol, ds, gamma, i);
% end

D_theta_J_i = G';
D2_theta_J_i = cell2mat(H);
dim_J = length(J);

% Get indicator
[L, D_J_L] = getIndicator(domain, ind_type, J, beta);

% Compute J(rho) and its derivative wrt rho
D_rho_L = D_J_L * D_theta_J_i * D_rho_theta_i;
T = D_theta_J_i*D_t_theta_i;
X = T'*T;
invTX = (inv(T'*T))';
V = sqrt(det(X));
Jr = L*V; %
Kr_1 = kron(eye(dim_t),T');
K_perm = permutationMatrix(dim_t, dim_t);
N = 0.5*(eye(dim_t^2)+K_perm);
Kr_2 = kron(D_t_theta_i',eye(dim_J));
Kr_3 = kron(eye(dim_t),D_theta_J_i);
D_r_Jr = zeros(dim_rho,1); %
for j = 1 : dim_rho
    D_r_Dtheta = diff(D_t_theta,rho(j));
    D_r_Dtheta_i = double(subs(D_r_Dtheta, [rho, t], [rho_learned, t_point]));
    D_r_Jr(j) = D_r_Jr(j) + ...
        D_rho_L(j) * V + ...
        L * V * transpose(invTX(:)) * N * Kr_1 * ...
        ( Kr_2 * D2_theta_J_i * D_rho_theta_i(:,j) + Kr_3 * D_r_Dtheta_i(:) );
end

end
