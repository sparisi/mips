%% Init
clear all
close all
reset(symengine)

utopia = 1e-8*[1,1]; % 0 would cause numerical problems
antiutopia = [350,350];


%% Parse settings 
indicator = {'utopia'};
indicator = {'antiutopia'};
indicator = {'pareto'};
indicator = {'mix1', 1.5}; % MIX1: I_AU * (1 - lambda * I_P)
indicator = {'mix2', [3,1]}; % MIX2: beta1 * I_AU / I_U - beta2
indicator = {'mix3', 1}; % MIX3: I_AU * (1 - lambda * I_U)
param_type = 'NN';

[J, theta, rho, t, D_theta_J, D2_theta_J, D_t_theta, D_rho_theta, I, I_params, AUp, Up] = ...
    settings_quadratic( indicator{1} );

switch indicator{1}
    case 'utopia' , params = num2cell(utopia,1);
    case 'antiutopia' , params = num2cell(antiutopia,1);
    case 'pareto' , params = {};
    case 'mix1' , params = num2cell([antiutopia, indicator{2}],1);
    case 'mix2' , params = num2cell([antiutopia, utopia, indicator{2}],1);
    case 'mix3' , params = num2cell([antiutopia, utopia, indicator{2}],1);
    otherwise , error('Unknown indicator.')
end

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

L = I*V;

% Full gradient of L(rho)
D_L = transpose(jacobian(L,rho));

% % Decomposed gradient of L(rho)
% D_rho_I = jacobian(I,rho);
% Kr_1 = kron(eye(dim_t),transpose(T));
% K_perm = permutationmatrix(dim_t,dim_t);
% N = 0.5*(eye(dim_t^2)+K_perm);
% Kr_2 = kron(transpose(D_t_theta),eye(dim_J));
% Kr_3 = kron(eye(dim_t),D_theta_J);
% D_L = sym('D_L',[dim_rho,1]);
% for i = 1 : dim_rho
%     D_r_Dtheta = diff(D_t_theta,rho(i));
%     D_L(i) = D_rho_I(i) * V + ...
%         I * V * transpose(invTX(:)) * N * Kr_1 * ...
%         ( Kr_2 * D2_theta_J * D_rho_theta(:,i) + Kr_3 * D_r_Dtheta(:) );
% end

D_L_fun = matlabFunction(D_L);
L_fun = matlabFunction(L);


%% Reset learning
n_points = 1024/2; % points for the Monte-Carlo estimate of the intregrals
n_points_plot = 100;
points = linspace(0,1,n_points);
tolerance = 0.01;

% rho_learned = zeros(1, dim_rho);
rho_learned = rand(1, dim_rho);
rho_history = [];
L_history = [];
iter = 1;
lrate = 1;


%% Learning
while true

    rho_arg = num2cell(rho_learned,1);
    D_L_eval = trapz(points,D_L_fun(params{:},rho_arg{:},points)');
    L_eval = trapz(points,L_fun(params{:},rho_arg{:},points));

    fprintf('Iter: %d, L: %.4f, Norm: %.3f\n', iter, L_eval, norm(D_L_eval));

    if norm(D_L_eval) < tolerance || lrate < 1e-8
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
            
            lambda = sqrt(D_L_eval * eye(length(D_L_eval')) * D_L_eval' / (4 * lrate));
            lambda = max(lambda,1e-8); % to avoid numerical problems
            stepsize = 1 / (2 * lambda);
            
            rho_learned = rho_learned + D_L_eval * stepsize;
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
points = linspace(0,1,n_points_plot);
figure
for j = 1 : size(rho_history,1)
    clf, hold all
    rho_arg = num2cell(rho_history(j,:));
    theta_learned = theta_fun(rho_arg{:},points);
    plot(theta_learned(1,:),theta_learned(2,:),'r')
    xlabel '\theta_1'
    ylabel '\theta_2'
    title 'Frontier in the Policy Parameters Space'
    drawnow
end


%% Plot front in the objectives space over time
J_fun = matlabFunction(J);
points = linspace(0,1,n_points_plot);
figure
for j = 1 : size(rho_history,1)
    clf, hold all
    rho_arg = num2cell(rho_history(j,:));
    front_learned = J_fun(rho_arg{:},points');
    plot(front_learned(:,1),front_learned(:,2),'r+')
    xlabel 'J_1'
    ylabel 'J_2'
    title 'Frontier in the Objectives Space'
    drawnow
end

autolayout
