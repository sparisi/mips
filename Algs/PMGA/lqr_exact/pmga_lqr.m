% Learning: sampled
% Evaluation: exact

%% Init
clc
clear all
reset(symengine)

domain = 'lqr';
param_type = 'P1';

[true_front, ~, utopia, antiutopia] = feval([domain '_moref'],0);
dim_J = feval([domain '_settings']);

[theta, rho, t, D_t_theta, D_rho_theta, J_sym] = params_lqr( param_type );

dim_rho = length(rho);
dim_t = length(t);
J_fun = matlabFunction(J_sym);

if dim_J == 2
    hypervfun = @(varargin)hypervolume2d(varargin{:},antiutopia,utopia);
else
    hypervfun = @(varargin)mexHypervolume(varargin{:},antiutopia,utopia,1e6);
end


%% Settings
loss_type = 'mix3';
% beta = [1, 1]; % MIX2: beta1 * L_AU / L_U - beta2
beta = 1;      % MIX3: L_AU * (1 - w * L_U)

tolerance = 0.00001;
lrate = 0.1;

n_points = 5; % #points t used to estimate the integral
lo = zeros(dim_t,1);
hi = ones(dim_t,1);
if dim_t == 1
    volume = 1;
else
    pp = zeros(dim_t+1, dim_t);
    for i = 1 : dim_t
        p = lo';
        p(i) = hi(i);
        pp(i,:) = p;
    end
    pp(end,:) = lo';
    [~, volume] = convhulln(pp);
end

MAX_ITER = 1000;

%% Reset learning
rho_learned = [1 1 0 0];
% rho_learned = [3 7];
rho_learned = rand(1,dim_rho);

rho_history = [];
Jr_history = [];
hv_history = [];
loss_history = [];
t_history = {};
iter = 0;

n_points_eval = 1000;


%% Learning
while iter < MAX_ITER
    
    iter = iter + 1;
    rho_history = [rho_history; rho_learned];
    
    theta_iter = subs(theta, rho, rho_learned);
    D_t_theta_iter = subs(D_t_theta, rho, rho_learned);
    D_rho_theta_iter = subs(D_rho_theta, rho, rho_learned);
    
    %%% Evaluation
    points_eval = samplePoints(lo, hi, n_points_eval, 1, 1);
    rho_arg = num2cell(rho_learned);
    t_arg = mat2cell(points_eval', size(points_eval,2), ones(1,dim_t));
    front_learned = -J_fun(rho_arg{:},t_arg{:});
    front_learned = pareto(front_learned);
    loss = eval_loss(front_learned, domain);
    hv = hypervfun(front_learned);
    loss_history = [loss_history; loss];
    hv_history = [hv_history; hv];
    fprintf('%d ) Loss: %.4f, Hyperv: %.4f\n', iter, loss, hv);
    %%%

    % Monte-Carlo sampling, estimate of the integral
    t_points = samplePoints(lo, hi, n_points, 1, 0)';
    t_history{iter} = t_points;
    D_jr_eval = zeros(n_points, dim_rho);
    Jr_eval = zeros(n_points, 1);
    for i = 1 : size(t_points,1)
        [Jr_i, D_jr_i] = executeTPoint(domain, t_points(i,:), theta_iter, D_t_theta, D_t_theta_iter, D_rho_theta_iter, t, rho, rho_learned, loss_type, beta);
        Jr_eval(i) = Jr_i;
        D_jr_eval(i,:) = D_jr_i;
    end
    Jr_eval = mean(Jr_eval,1) * volume;
    D_jr_eval = mean(D_jr_eval,1) * volume;
    
    % Update
%     fprintf('Iter: %d, Jr: %.4f, Norm: %.3f\n', iter, Jr_eval, norm(D_jr_eval));
    Jr_history = [Jr_history; Jr_eval];
    lambda = sqrt(D_jr_eval * eye(dim_rho) * D_jr_eval' / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
    rho_learned = rho_learned + D_jr_eval * stepsize;
    
end


%% Plot results
close all
points_eval = linspace(0,1,n_points_eval);
figure
for j = 1 : size(rho_history,1)
    clf, hold on
    plot(true_front(:,1),true_front(:,2))
    rho_arg = num2cell(rho_history(j,:));
    front_learned = -J_fun(rho_arg{:},points_eval');
    plot(front_learned(:,1),front_learned(:,2),'r')
    drawnow
end

scrsz = get(groot,'ScreenSize');
figure('Position',[1 scrsz(4)/2 scrsz(3)/2 scrsz(4)/2])

subplot(1,2,1,'align')
plot(loss_history)
title('Loss')

subplot(1,2,2)
plot(hv_history)
title('Hypervolume')
