%% Init
clc
clear all
reset(symengine)

domain = 'dam';
param_type = 'P1';

[~, ~, utopia, antiutopia] = feval([domain '_moref'],0);
[dim_J, pol, ~, ~, gamma] = feval([domain '_settings']);
[theta, rho, t, D_t_theta, D_rho_theta] = feval(['params_' domain], param_type);

dim_rho = length(rho);
dim_t = length(t);

D_r_Dtheta = cell(dim_rho,1);
for j = 1 : dim_rho
    D_r_Dtheta{j} = diff(D_t_theta,rho(j));
end

%% Settings
loss_type = 'mix2';
beta = [1, 1]; % MIX2: beta1 * L_AU / L_U - beta2
% beta = 1;      % MIX3: L_AU * (1 - w * L_U)

tolerance = 0.00001;
lrate = 2;

% #points t used to estimate the integral
n_points = 1;
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
% rho_learned = -20*ones(1,dim_rho);
rho_learned = rand(1, dim_rho);
rho_learned(end) = 50;

rho_history = [];
Jr_history = [];
hv_history = [];
loss_history = [];
t_history = {};
iter = 0;


%% Learning
while iter < MAX_ITER
    
    iter = iter + 1;
    rho_history = [rho_history; rho_learned];
    
    theta_iter = subs(theta, rho, rho_learned);
    D_t_theta_iter = subs(D_t_theta, rho, rho_learned);
    D_rho_theta_iter = subs(D_rho_theta, rho, rho_learned);
    
    % Monte-Carlo sampling, estimate of the integral
    t_points = samplePoints(lo, hi, n_points, 1, 0)';
    t_history{iter} = t_points;
    D_jr_eval = zeros(n_points, dim_rho);
    Jr_eval = zeros(n_points, 1);
    for i = 1 : size(t_points,1)
        [Jr_i, D_jr_i] = executeTPoint(domain, t_points(i,:), theta_iter, D_r_Dtheta, D_t_theta_iter, D_rho_theta_iter, t, rho, rho_learned, loss_type, beta);
        Jr_eval(i) = Jr_i;
        D_jr_eval(i,:) = D_jr_i;
    end
    Jr_eval = mean(Jr_eval,1) * volume;
    D_jr_eval = mean(D_jr_eval,1) * volume;
    
    % Update
    fprintf('Iter: %d, Jr: %.4f, Norm: %.3f\n', iter, Jr_eval, norm(D_jr_eval));
    Jr_history = [Jr_history; Jr_eval];
    
    lambda = sqrt(D_jr_eval * eye(dim_rho) * D_jr_eval' / (4 * lrate));
    lambda = max(lambda,1e-8); % to avoid numerical problems
    stepsize = 1 / (2 * lambda);
    
    rho_learned = rho_learned + D_jr_eval * stepsize;
    
end


%% Evaluation
if dim_J == 2
    hv_fun = @(varargin)hypervolume2d(varargin{:},antiutopia,utopia);
else
    hv_fun = @(varargin)hypervolume(varargin{:},antiutopia,utopia,1e6);
end

n_points_eval = 1000;
episodes_eval = 100;
steps_eval = 100;
t_points_eval = samplePoints(lo, hi, n_points_eval, 1, 1)';

close all
figure
for j = 1 : 10 : size(rho_history,1)
    
    clf, hold all
    
    theta_iter = subs(theta, rho, rho_history(j,:));
    front_learned = zeros(n_points_eval, dim_J);

    parfor i = 1 : size(t_points_eval,1)
        theta_i = double(subs(theta_iter, t, t_points_eval(i,:)));
        pol_iter = pol;
        pol_iter.theta = theta_i;
%         [~, J] = collect_samples(domain, episodes_eval, steps_eval, pol_iter);
        [~, J] = collect_samples_rele(domain, episodes_eval, steps_eval, pol_iter);
        front_learned(i,:) = J;
    end
    
    front_learned = sortrows(front_learned);
%     front_learned = pareto(front_learned);
    if dim_J == 3
        plot3(front_learned(:,1),front_learned(:,2),front_learned(:,3),'+r')
        grid on, view(70,24)
    else
        plot(front_learned(:,1),front_learned(:,2),'r')
    end
    front_manual = feval([domain '_moref'],1);
    
    front_learned = pareto(front_learned);
    hv(j) = hv_fun(front_learned);
    l(j) = eval_loss(front_learned,domain);
    fprintf('%d ) Hyperv: %.4f, Loss: %.4f\n',j,hv(j),l(j));

    drawnow
    
end
