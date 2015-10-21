%% Init
clear all
close all
clc
reset(symengine)

load('front_lqr3.mat');

loss_type = 'mix1';
param_type = 'P2';
mexName = strcat(['mexIntegrate_LQR3_' param_type '_' loss_type]);

[J, theta, rho, t] = settings_lqr3( loss_type, param_type );

dim_rho = size(rho,2); % k


%% Settings
utopia = [195,195,195];
antiutopia = [360,360,360];

loss_params = 30;       % MIX1: L_AU * (1 - w * L_pareto)
% loss_params = [2.5,1];  % MIX2: beta1 * L_AU / L_U - beta2
% loss_params = 1.4;      % MIX3: L_AU * (1 - w * L_U)

if strcmp(loss_type, 'utopia')
    params = utopia;
elseif strcmp(loss_type, 'antiutopia')
    params = antiutopia;
elseif strcmp(loss_type, 'pareto')
    params = {};
elseif strcmp(loss_type, 'mix1')
    params = [antiutopia, loss_params];
elseif strcmp(loss_type, 'mix2')
    params = [antiutopia, utopia, loss_params];
elseif strcmp(loss_type, 'mix3')
    params = [antiutopia, utopia, loss_params];
else
    error('Unknown loss function.')
end

n_points = 10000;
n_points_plot = 300;
useSimplex = 1;
tmin = [0,0];
tmax = [1,1];
tolerance = 0.01;


%% Reset learning
rho_learned = rand(1, dim_rho) / 1000;
% rho_learned = ones(1, dim_rho) / 1000;
% rho_learned = zeros(1, dim_rho);
rho_history = [];
Jr_history = [];
iter = 0;
lrate = 1;


%% Learning
while true

    iter = iter + 1;

    [Jr_eval, D_jr_eval] = feval(mexName, tmin, tmax, n_points, useSimplex, rho_learned, params);
    
    fprintf('Iter: %d, Jr: %.4f, Norm: %.3f\n', iter, Jr_eval, norm(D_jr_eval));
    
    if norm(D_jr_eval) < tolerance || lrate < 1e-12
        rho_history = [rho_history; rho_learned];
        Jr_history = [Jr_history; Jr_eval];
        break
    elseif sum(isnan(D_jr_eval)) > 0
        break
    else
        % If the performance decreased, revert and halve the learning rate
        if ~isempty(Jr_history) && Jr_eval < Jr_history(end)
            fprintf('Reducing learning rate...\n')
            lrate = lrate / 2;
            rho_learned = rho_history(end,:);
            iter = iter - 1;
        else
            rho_history = [rho_history; rho_learned];
            Jr_history = [Jr_history; Jr_eval];
            
            lambda = sqrt(D_jr_eval' * eye(length(D_jr_eval)) * D_jr_eval / (4 * lrate));
            lambda = max(lambda,1e-8); % to avoid numerical problems
            stepsize = 1 / (2 * lambda);
            
            rho_learned = rho_learned + D_jr_eval' * stepsize;
        end
    end            
    
end


%% Plot J(rho) over time
figure
plot(Jr_history)
xlabel 'Iterations'
ylabel 'J(\rho)'


%% Plot front in parameter space over time
theta_fun = matlabFunction(theta);
points = samplePoints([0,0]',[1,1]',n_points_plot,useSimplex,1);
p_t1 = points(1,:);
p_t2 = points(2,:);
figure
for j = 1 : size(rho_history,1)
    clf, hold all
    scatter3(k1_opt,k2_opt,k3_opt)
    view(60,54)
    rho_arg = num2cell(rho_history(j,:));
    theta_learned = theta_fun(rho_arg{:},p_t1,p_t2)';
    scatter3(theta_learned(:,1),theta_learned(:,2),theta_learned(:,3),'r')
    xlabel '\theta_1'
    ylabel '\theta_2'
    zlabel '\theta_3'
    title 'Frontier in the Policy Parameters Space'
    drawnow
end


%% Plot front in objective space over time
J_fun = matlabFunction(J);
points = samplePoints([0,0]', [1,1]', n_points_plot, useSimplex, 1);
p_t1 = points(1,:);
p_t2 = points(2,:);
figure
for j = 1 : size(rho_history,1)
    clf, hold all
    scatter3(front_manual(:,1),front_manual(:,2),front_manual(:,3))
    view(144,22)
    rho_arg = num2cell(rho_history(j,:));
    front_learned = J_fun(rho_arg{:},p_t1',p_t2');
    scatter3(front_learned(:,1),front_learned(:,2),front_learned(:,3),'r')
    xlabel 'J_1'
    ylabel 'J_2'
    zlabel 'J_3'
    title 'Frontier in the Objectives Space'
    drawnow
end
