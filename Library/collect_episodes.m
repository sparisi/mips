function [J, Theta] = collect_episodes(domain, maxepisodes, pol_high)
% Collects episodes for the specified domain. The low-level policy is
% deterministic and its parameters are drawn from a high-level
% distribution (pol_high).

[n_obj, pol_low, ~, steps] = feval([domain '_settings']);
pol_low = pol_low.makeDeterministic;

J = zeros(maxepisodes,n_obj);
dim_theta = pol_high.dim;
Theta = zeros(dim_theta,maxepisodes);

parfor k = 1 : maxepisodes
    
    % Draw theta from the high-level policy and perform a rollout
    pol_tmp = pol_low;
    theta = pol_high.drawAction;
    pol_tmp.theta(1:dim_theta) = theta;
    Theta(:,k) = theta;

%     [~, J_ep] = collect_samples(domain, 1, steps, pol_tmp);
    [~, J_ep] = collect_samples_rele(domain, 1, steps, pol_tmp);

    J(k,:) = J_ep;
    
end

end
