%% Init
clear all
close all
clc
reset(symengine)

load('front_lqr2.mat');

loss_type = 'mix3';
param_type = 'P2';

[J, theta, rho, t, D_theta_J, D2_theta_J, D_t_theta, D_rho_theta, L, L_params, AUp, Up] = ...
    settings_lqr2( loss_type, param_type );

dim_J = size(J,2);     % q
dim_rho = size(rho,2); % k
dim_t = size(t,2);     % b


%% Derivatives
T = D_theta_J*D_t_theta;
X = transpose(T)*T;
detX = X; % since it is a scalar
invX = 1 / X;
invTX = transpose(invX);
V = sqrt(detX);

Jr = L*V;

% Full gradient of J(rho)
D_jr = transpose(jacobian(Jr,rho));

% % Decomposed gradient of J(rho)
% D_rho_L = jacobian(L,rho);
% Kr_1 = kron(eye(dim_t),transpose(T));
% K_perm = permutationMatrix(dim_t,dim_t);
% N = 0.5*(eye(dim_t^2)+K_perm);
% Kr_2 = kron(transpose(D_t_theta),eye(dim_J));
% Kr_3 = kron(eye(dim_t),D_theta_J);
% D_jr = sym('D_jr',[dim_rho,1]);
% for i = 1 : dim_rho
%     D_r_Dtheta = diff(D_t_theta,rho(i));
%     D_jr(i) = D_rho_L(i) * V + ...
%         L * V * transpose(invTX(:)) * N * Kr_1 * ...
%         ( Kr_2 * D2_theta_J * D_rho_theta(:,i) + Kr_3 * D_r_Dtheta(:) );
% end

D_jr_fun = matlabFunction(D_jr);
Jr_fun = matlabFunction(Jr);


%% Settings
utopia = [150,150];
antiutopia = [310,310];

% loss_params = 1.5;    % MIX1: L_AU * (1 - w * L_pareto)
% loss_params = [1,1];  % MIX2: beta1 * L_AU / L_U - beta2
loss_params = 1;      % MIX3: L_AU * (1 - w * L_U)

if strcmp(loss_type, 'utopia')
    params = num2cell(utopia,1);
elseif strcmp(loss_type, 'antiutopia')
    params = num2cell(antiutopia,1);
elseif strcmp(loss_type, 'pareto')
    params = {};
elseif strcmp(loss_type, 'mix1')
    params = num2cell([antiutopia, loss_params],1);
elseif strcmp(loss_type, 'mix2')
    params = num2cell([antiutopia, utopia, loss_params],1);
elseif strcmp(loss_type, 'mix3')
    params = num2cell([antiutopia, utopia, loss_params],1);
else
    error('Unknown loss function.')
end

n_points = 1024/2;
n_points_plot = 100;
points = linspace(0,1,n_points);
tolerance = 0.01;


%% Reset learning
% rho_learned = [3 7]; % quite far
rho_learned = [12 12]; % symmetric
% rho_learned = [1 1 0 0];
% rho_learned = zeros(1, dim_rho);
rho_history = [];
Jr_history = [];
iter = 0;
lrate = 1;


%% Learning
while true

    iter = iter + 1;

    rho_arg = num2cell(rho_learned,1);
    D_jr_eval = trapz(points,D_jr_fun(params{:},rho_arg{:},points)');
    Jr_eval = trapz(points,Jr_fun(params{:},rho_arg{:},points));

    fprintf('Iter: %d, Jr: %.4f, Norm: %.3f\n', iter, Jr_eval, norm(D_jr_eval));

    if norm(D_jr_eval) < tolerance || lrate < 1e-3
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
            
            lambda = sqrt(D_jr_eval * eye(length(D_jr_eval')) * D_jr_eval' / (4 * lrate));
            lambda = max(lambda,1e-8); % to avoid numerical problems
            stepsize = 1 / (2 * lambda);
            
            rho_learned = rho_learned + D_jr_eval * stepsize;
        end
    end            
    
end


%% Plot J(rho) over time
figure
plot(Jr_history)


%% Plot theta over time
theta_fun = matlabFunction(theta);
n_k = length(k1_opt);
points = linspace(0,1,n_k);
x_axis = (1:n_k)/n_k;
figure
for j = 1 : size(rho_history,1)
    clf, hold all
    plot(x_axis,k1_opt,'r')
    plot(x_axis,k2_opt,'b')
    rho_arg = num2cell(rho_history(j,:));
    theta_learned = theta_fun(rho_arg{:},points);
    plot(x_axis,theta_learned(1,:),'-.r')
    plot(x_axis,theta_learned(2,:),'-.b')
    axis([0,1,-1,0])
    drawnow
end


%% Plot front in parameter space over time
theta_fun = matlabFunction(theta);
points = linspace(0,1,n_points_plot);
figure
for j = 1 : size(rho_history,1)
    clf, hold all
    plot(k1_opt,k2_opt)
    rho_arg = num2cell(rho_history(j,:));
    theta_learned = theta_fun(rho_arg{:},points);
    plot(theta_learned(1,:),theta_learned(2,:),'r')
    drawnow
end


%% Plot front in objective space over time
J_fun = matlabFunction(J);
points = linspace(0,1,n_points_plot);
figure
for j = 1 : size(rho_history,1)
    clf, hold all
    plot(front_manual(:,1),front_manual(:,2))
    rho_arg = num2cell(rho_history(j,:));
    front_learned = J_fun(rho_arg{:},points');
    plot(front_learned(:,1),front_learned(:,2),'r')
    drawnow
end


% %% Plot J(rho) (only if rho is a 2d vector)
% [xg,yg] = meshgrid(-2:0.2:2, -2:0.2:2);
% VV = [xg(:), yg(:)];
% F = zeros(length(VV),3);
% points = linspace(0,1,100);
% for i = 1 : length(VV)
%     D_jr_eval = Jr_fun(params{:},VV(i,1),VV(i,2),points);
%     z = trapz(points,D_jr_eval');
%     F(i,:) = [VV(i,1),VV(i,2),z];
% end
% figure; surf(xg,yg,reshape(F(:,3),size(xg,1),size(yg,2)));
% xlabel('\rho_1')
% ylabel('\rho_2')
% zlabel('J_\rho')
% [I,II] = max(F(:,3));
% F(II,:)
