%% Init
clear all
close all
reset(symengine)
rng(1)

dim = 3;

% parameterization: P1 constrained, P2 unconstrained, NN neural net

mdp = Dam(dim); bfs = @dam_basis_rbf; mu0 = [50, -50, 0, 0, 50]; sigma0 = 150^2;
[theta, rho, t, D_t_theta, D_rho_theta] = feval(['params_' lower(class(mdp))], [], mdp.dreward);
policy = GaussianLinearDiag(bfs, mdp.daction, mu0, sigma0);

% mdp = LQR_MO(dim); bfs = @(varargin)basis_poly(1,dim,0,varargin{:}); mu0 = -0.5*eye(dim); sigma0 = eye(dim);
% [theta, rho, t, D_t_theta, D_rho_theta] = feval(['params_' lower(class(mdp))], 'P1', mdp.dreward);
% policy = GaussianLinearFixedvarDiagmean(bfs, mdp.daction, mu0, sigma0);

utopia = mdp.utopia;
antiutopia = mdp.antiutopia;
true_front = mdp.truefront;
dim_J = mdp.dreward;

ind_type = {'mix2', [1,1]}; % MIX2: beta1 * I_AU / I_U - beta2
% ind_type = {'mix3', 1}; % MIX3: I_AU * (1 - lambda * I_U)
ind_type = {'hv', [1,1]}; % Hypervolume contribution
[ind_handle, ind_d_handle] = parse_indicator_handle(ind_type{1},ind_type{2},utopia,antiutopia);

dim_rho = length(rho);
dim_t = length(t);
dim_theta = length(theta);

for j = 1 : dim_rho
    D_r_Dtheta(:,:,j) = diff(D_t_theta,rho(j));
end
D_r_Dtheta = reshape(D_r_Dtheta,dim_theta*dim_t,dim_rho);


%% Learning settings
tolerance = 0.00001;
lrate = 0.1;
optim = ADAM(dim_rho);
optim.alpha = lrate;

episodes = 50;
steps = 50;
n_points = 15; % #points t used to estimate the integral
lo = zeros(dim_t,1);
hi = ones(dim_t,1);
[~, volume] = simplex(lo,hi);

MAX_ITER = 1000;


%% Init learning
% rho_learned = -20*ones(1,dim_rho);
% rho_learned = rand(1, dim_rho) - 0.5;
rho_learned = (rand(1, dim_rho) - 0.5) * 5;
% rho_learned(end) = 50;

rho_history = [];
L_history = [];
hv_history = [];
loss_history = [];
t_history = {};
iter = 1;


%% Learning
while iter < MAX_ITER
    
    rho_history = [rho_history; rho_learned];
    
    theta_iter = subs(theta, rho, rho_learned);
    D_t_theta_iter = subs(D_t_theta, rho, rho_learned);
    D_rho_theta_iter = subs(D_rho_theta, rho, rho_learned);
    D_r_Dtheta_iter = subs(D_r_Dtheta, rho, rho_learned);
    
    % Monte-Carlo sampling, estimate of the integral
    t_points = unifrnds(lo, hi, n_points)';
    t_history{iter} = t_points;

    [L_eval, D_L_eval] = executeTPoint(mdp, policy, episodes, steps, t_points, ...
        theta_iter, D_r_Dtheta_iter, D_t_theta_iter, D_rho_theta_iter, ...
        t, ind_handle, ind_d_handle);

    L_eval = L_eval * volume;
    D_L_eval = D_L_eval * volume;
    
    % Update
    fprintf('%d) Loss (learn): %.4f, Norm: %.3f\n', iter, L_eval, norm(D_L_eval));
    L_history = [L_history; L_eval];
%     rho_learned = rho_learned + D_L_eval / norm(D_L_eval) * lrate;
    rho_learned = optim.step(rho_learned, -D_L_eval);
    
    iter = iter + 1;

end


%% Evaluation
if dim_J == 2
    hypervfun = @(varargin)hypervolume2d(varargin{:},antiutopia,utopia);
else
%     hypervfun = @(varargin)mexHypervolume(varargin{:},antiutopia,utopia,1e6);
    hypervfun = @(varargin)hypervolume(varargin{:},antiutopia,utopia,1e6);
end
hv_ref = hypervfun(true_front);

n_points_eval = 100;
t_points_eval = linspacesim(lo, hi, n_points_eval);
t_arg = mat2cell(t_points_eval', size(t_points_eval,2), ones(1,dim_t));
episodes_eval = 100;
steps_eval = 100;
front_manual = mdp.truefront;
theta_f = matlabFunction(theta);

close all
figure


%% Plotting
for j = 1 : 50 : size(rho_history,1)
    
    clf, hold all

    rho_arg = num2cell(rho_history(j,:));
    theta_j = reshape(theta_f(rho_arg{:},t_arg{:}), [size(t_points_eval,2), dim_theta]);
    pol_iter = repmat(policy,1,size(t_points_eval,2));
    for i = 1 : size(t_points_eval,2)
        pol_iter(i) = policy.update(theta_j(i,:));
    end
    
    front_learned = evaluate_policies(mdp, episodes_eval, steps_eval, pol_iter)';
    front_learned = sortrows(front_learned);
%     front_learned = pareto(front_learned);
    if dim_J == 3
        plot3(front_manual(:,1),front_manual(:,2),front_manual(:,3),'ob')
        plot3(front_learned(:,1),front_learned(:,2),front_learned(:,3),'+r')
        grid on, view(70,24)
    else
        plot(front_manual(:,1),front_manual(:,2),'b')
        plot(front_learned(:,1),front_learned(:,2),'r')
    end
    
    l(j) = sum(ind_handle(front_learned));
    front_learned = pareto(front_learned);
    hv(j) = hypervfun(front_learned);
    fprintf('%d ) Hyperv: %.4f / %.4f, Loss: %.4f\n',j,hv(j),hv_ref,l(j));

    title(num2str(j))
    drawnow limitrate
    
end
