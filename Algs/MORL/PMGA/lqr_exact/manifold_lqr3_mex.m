%% Init
clear, clear global
close all
clc
reset(symengine)

load('front_lqr3.mat');
utopia = [195,195,195];
antiutopia = [360,360,360];


%% Parse settings 
indicator = {'utopia'};
% indicator = {'antiutopia'};
% indicator = {'pareto'};
% indicator = {'mix1', 30}; % MIX1: I_AU * (1 - lambda * I_P)
% indicator = {'mix2', [2.5,1]}; % MIX2: beta1 * I_AU / I_U - beta2
% indicator = {'mix3', 1.4}; % MIX3: I_AU * (1 - lambda * I_U)
param_type = 'P1'; % P1 unconstrained, P2 constrained

mexName = strcat(['mexIntegrate_LQR3_' param_type '_' indicator{1}]);

switch indicator{1}
    case 'utopia' , params = utopia;
    case 'antiutopia' , params = antiutopia;
    case 'pareto' , params = [];
    case 'mix1' , params = [antiutopia, indicator{2}];
    case 'mix2' , params = [antiutopia, utopia, indicator{2}];
    case 'mix3' , params = [antiutopia, utopia, indicator{2}];
    otherwise , error('Unknown indicator.')
end

[J, theta, rho, t] = settings_lqr3( indicator{1}, param_type );
dim_rho = size(rho,2); % k


%% Reset learning
n_points = 10000;
n_points_plot = 300;
tmin = [0,0];
tmax = [1,1];
tolerance = 0.01;

rng(4)
rho_learned = rand(1, dim_rho) / 1000;
% rho_learned = ones(1, dim_rho);
% rho_learned = zeros(1, dim_rho);
rho_history = [];
L_history = [];
iter = 1;
lrate = 1;


%% Learning
while true

    [L_eval, D_L_eval] = feval(mexName, tmin, tmax, n_points, 1, rho_learned, params);
    
    fprintf('Iter: %d, L: %.4f, Norm: %.3f\n', iter, L_eval, norm(D_L_eval));
    
    if norm(D_L_eval) < tolerance || lrate < 1e-12
        rho_history = [rho_history; rho_learned];
        L_history = [L_history; L_eval];
        break
    elseif any(isnan(D_L_eval)) > 0
        break
    else
        % If the performance decreased, revert and halve the learning rate
        if ~isempty(L_history) && L_eval < L_history(end)
            fprintf('Reducing learning rate...\n')
            lrate = lrate / 2;
            rho_learned = rho_history(end,:);
        else
            rho_history = [rho_history; rho_learned];
            L_history = [L_history; L_eval];
            
            lambda = sqrt(D_L_eval' * eye(length(D_L_eval)) * D_L_eval / (4 * lrate));
            lambda = max(lambda,1e-8); % to avoid numerical problems
            stepsize = 1 / (2 * lambda);
            
            rho_learned = rho_learned + D_L_eval' * stepsize;
            iter = iter + 1;
        end
    end            

end


%% Plot L(rho) over time
figure
plot(L_history)
xlabel 'Iterations'
ylabel 'L(\rho)'


%% Plot front in the parameters space over time
theta_fun = matlabFunction(theta);
points = linspacesim([0,0]',[1,1]',n_points_plot);
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


%% Plot front in objectives space over time
J_fun = matlabFunction(J);
points = linspacesim([0,0]', [1,1]', n_points_plot);
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

autolayout
