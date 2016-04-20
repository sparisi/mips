function [L, D_L] = executeTPoint(mdp, policy, episodes, steps, t_points, ...
    theta_iter, D_r_Dtheta_iter, D_t_theta_iter, D_rho_theta_iter, ...
    t, I_handle, D_I_handle)

n_obj = mdp.dreward;
gamma = mdp.gamma;

[n_points, dim_t] = size(t_points);

L = 0;
D_L = 0;

parfor i = 1 : n_points

    % Get closed-form derivatives
    theta_i = double(subs(theta_iter, t, t_points(i,:)));
    D_t_theta_i = double(subs(D_t_theta_iter, t, t_points(i,:)));
    D_rho_theta_i = double(subs(D_rho_theta_iter, t, t_points(i,:)));
    D_r_Dtheta_i = double(subs(D_r_Dtheta_iter, t, t_points(i,:)));

    policy_i = policy.update(theta_i);
    
    % Collect samples and estimate gradients and hessians
    [ds, J] = collect_samples(mdp, episodes, steps, policy_i);
    D_theta_J_i = GPOMDPbase(policy_i, ds, gamma)';
    D2_theta_J_i = HessianRFbase(policy_i, ds, gamma);
    D2_theta_J_i = D2_theta_J_i(:,:)';
    
    % Get indicator
    I = I_handle(J');
    D_J_I = D_I_handle(J');

    % Compute L(rho) and its derivative wrt rho
    D_rho_I = D_J_I * D_theta_J_i * D_rho_theta_i;
    T = D_theta_J_i*D_t_theta_i;
    X = T'*T;
    invTX = (inv(T'*T))';
    V = sqrt(det(X));
    L = L + I*V;
    Kr_1 = kron(eye(dim_t),T');
    K_perm = permutationmatrix(dim_t, dim_t);
    N = 0.5*(eye(dim_t^2)+K_perm);
    Kr_2 = kron(D_t_theta_i',eye(n_obj));
    Kr_3 = kron(eye(dim_t),D_theta_J_i);

    D_L = D_L + D_rho_I * V + ...
        I * V * transpose(invTX(:)) * N * Kr_1 * ...
        ( Kr_2 * D2_theta_J_i * D_rho_theta_i + Kr_3 * D_r_Dtheta_i );
    
end

L = L / n_points;
D_L = D_L / n_points;
