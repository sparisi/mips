% Learning: sampled
% Evaluation: exact

%% Init
clear all
reset(symengine)

rng(1)

dim = 2;
mdp = LQR_MO(dim);
policy = GaussianLinearFixedvarDiagmean( ...
    @(varargin)basis_poly(1,dim,0,varargin{:}), ...
    dim, -0.5*eye(dim), eye(dim));
utopia = mdp.utopia;
antiutopia = mdp.antiutopia;
true_front = mdp.truefront;
dim_J = mdp.dreward;

param_type = 'P1'; % P1 unconstrained, P2 constrained, NN neural network
ind_type = {'mix2', [1,1]}; % MIX2: beta1 * I_AU / I_U - beta2
ind_type = {'mix3', 1}; % MIX3: I_AU * (1 - lambda * I_U)
[ind_handle, ind_d_handle] = parse_indicator_handle(ind_type{1},ind_type{2},utopia,antiutopia);

[theta, rho, t, D_t_theta, D_rho_theta, J_sym] = params_lqr_mo( param_type, mdp.dreward );
dim_rho = length(rho);
dim_t = length(t);
dim_theta = length(theta);
J_fun = matlabFunction(J_sym);

for j = 1 : dim_rho
    D_r_Dtheta(:,:,j) = diff(D_t_theta,rho(j));
end
D_r_Dtheta = reshape(D_r_Dtheta,dim_theta*dim_t,dim_rho);

if dim_J == 2
    hypervfun = @(varargin)hypervolume2d(varargin{:},antiutopia,utopia);
else
%     hypervfun = @(varargin)mexHypervolume(varargin{:},antiutopia,utopia,1e5);
    hypervfun = @(varargin)hypervolume(varargin{:},antiutopia,utopia,1e5);
end
hv_ref = hypervfun(true_front);

%% Learning settings
tolerance = 0.00001;
lrate = 0.01;
optim = ADAM(dim_rho);
optim.alpha = lrate;

episodes = 50;
steps = 50;
n_points = 10; % #points t used to estimate the integral
n_points_eval = 500;
lo = -ones(dim_t,1);
hi = ones(dim_t,1);
[~, volume] = simplex(lo,hi);

MAX_ITER = 10000;


%% Init learning
rho_learned = [1 1 0 0];
% rho_learned = [3 7];
rho_learned = (rand(1,dim_rho)-0.5)*1e-3;

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
    
    %% Evaluation
    points_eval = linspacesim(lo, hi, n_points_eval);
    rho_arg = num2cell(rho_learned);
    t_arg = mat2cell(points_eval', size(points_eval,2), ones(1,dim_t));
    front_learned = -J_fun(rho_arg{:},t_arg{:});
    loss = sum(ind_handle(front_learned));

    hv = hypervfun(front_learned);
    loss_history = [loss_history; loss];
    hv_history = [hv_history; hv];
    fprintf('%d ) Loss: %.4f, Hyperv: %.4f / %.4f', iter, loss, hv, hv_ref);

    %% Monte-Carlo sampling, estimate of the integral
    t_points = unifrnds(lo, hi, n_points)';
    t_history{iter} = t_points;

    [L_eval, D_L_eval] = executeTPoint(mdp, policy, episodes, steps, t_points, ...
        theta_iter, D_r_Dtheta_iter, D_t_theta_iter, D_rho_theta_iter, ...
        t, ind_handle, ind_d_handle);

    L_eval = L_eval * volume;
    D_L_eval = D_L_eval * volume;
    
    %% Update
    fprintf(', Loss (learn): %.4f, Norm: %.3f', L_eval, norm(D_L_eval));
    fprintf('\n');
    L_history = [L_history; L_eval];
%     rho_learned = rho_learned + D_L_eval / norm(D_L_eval) * lrate;
    rho_learned = optim.step(rho_learned, -D_L_eval);
    
    iter = iter + 1;

end


%% Plot results
close all
points_eval = linspacesim(lo, hi, n_points_eval);
figure
for j = [1 : 1 : size(rho_history,1), size(rho_history,1)]
    clf, hold on
    rho_arg = num2cell(rho_history(j,:));
    t_arg = mat2cell(points_eval', size(points_eval,2), ones(1,dim_t));
    front_learned = -J_fun(rho_arg{:},t_arg{:});
    if dim == 2
        plot(true_front(:,1),true_front(:,2),'linewidth',4)
        plot(front_learned(:,1),front_learned(:,2),'r','linewidth',2)
    elseif dim == 3
        grid on, view(70,24)
        scatter3(true_front(:,1),true_front(:,2),true_front(:,3))
        scatter3(front_learned(:,1),front_learned(:,2),front_learned(:,3),'r')
    end
    title(num2str(j))
    drawnow
    pause(0.001)
end

scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])

subplot(1,2,1,'align')
plot(loss_history)
title('Loss')

subplot(1,2,2)
plot(hv_history)
title('Hypervolume')

autolayout