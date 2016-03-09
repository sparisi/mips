function [Jr, D_r_Jr] = executeTPoint(mdp, policy, episodes, steps, t_points, ...
    theta_iter, D_r_Dtheta_iter, D_t_theta_iter, D_rho_theta_iter, ...
    t, indicator, D_indicator)

n_obj = mdp.dreward;
gamma = mdp.gamma;

[n_points, dim_t] = size(t_points);

Jr = 0;
D_r_Jr = 0;

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
    L = indicator(J');
    D_J_L = D_indicator(J');

    % Compute J(rho) and its derivative wrt rho
    D_rho_L = D_J_L * D_theta_J_i * D_rho_theta_i;
    T = D_theta_J_i*D_t_theta_i;
    X = T'*T;
    invTX = (inv(T'*T))';
    V = sqrt(det(X));
    Jr = Jr + L*V;
    Kr_1 = kron(eye(dim_t),T');
    K_perm = permutationmatrix(dim_t, dim_t);
    N = 0.5*(eye(dim_t^2)+K_perm);
    Kr_2 = kron(D_t_theta_i',eye(n_obj));
    Kr_3 = kron(eye(dim_t),D_theta_J_i);

    D_r_Jr = D_r_Jr + D_rho_L * V + ...
        L * V * transpose(invTX(:)) * N * Kr_1 * ...
        ( Kr_2 * D2_theta_J_i * D_rho_theta_i + Kr_3 * D_r_Dtheta_i );
    
end

Jr = Jr / n_points;
D_r_Jr = D_r_Jr / n_points;
