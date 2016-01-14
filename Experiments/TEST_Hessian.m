%% Init
clear all
clc

dim = 2;
mdp   = LQR(dim);
robj = 1;
gamma = mdp.gamma;
Q     = mdp.Q;
R     = mdp.R;
x0    = mdp.x0;

A0 = -0.5 * eye(dim);
Sigma = eye(dim);
bfs = @(varargin)basis_poly(1,dim,0,varargin{:});
policy = GaussianLinearFixedvarDiagmean(bfs, dim, A0, Sigma);


%% Compute closed forms
reset(symengine)

J = sym('J',[dim,1]);
K = sym('k',[dim,dim]);
k1_2 = 0; k2_1 = 0; K = subs(K);
theta = nonzeros(K);
dtheta = length(theta);

for i = 1 : dim
    P = (Q{i}+K*R{i}*K)*(eye(dim)-gamma*(eye(dim)+2*K+K^2))^-1; % Only when A = B = I
    J(i) = -transpose(x0)*P*x0 + (1/(1-gamma))*trace(Sigma*(R{i}+gamma*P));
end

J = J(robj);
D_theta_J = transpose(jacobian(J,theta));
H_theta_J = hessian(J,theta);

H_ex = double(subs(H_theta_J,theta,policy.theta));
G_ex = double(subs(D_theta_J,theta,policy.theta));
J_ex = double(subs(J,theta,policy.theta));


%% Run
trials = 10;
episodes = 200;
steps = 50;
gamma = mdp.gamma;

for i = 1 : trials
    
    [ds, J_i] = collect_samples(mdp, episodes, steps, policy);
    H_rf = HessianRF(policy, ds, gamma);
    H_rfb = HessianRFbase(policy, ds, gamma);
    G_gpomdp = GPOMDP(policy, ds, gamma);
    G_gpomdpb = GPOMDPbase(policy, ds, gamma);
    
    H(:,:,i) = H_rf(:,:,robj);
    Hbase(:,:,i) = H_rfb(:,:,robj);

    G(:,i) = G_gpomdp(:,robj);
    Gbase(:,i) = G_gpomdpb(:,robj);
    
    J_est(i) = J_i(robj);
    
end

%% Hessian
fprintf('======= STD - HESSIAN ======= \n')
disp(num2str(std(H,1,3)));
fprintf('======= STD - HESSIAN WITH BASELINE ======= \n')
disp(num2str(std(Hbase,1,3)));

err = abs(H_ex - mean(H,3)).^2;
MSE_h = sum(err(:)) / numel(H_ex);
err = abs(H_ex - mean(Hbase,3)).^2;
MSE_hbase = sum(err(:)) / numel(H_ex);

fprintf('\n')
fprintf('MSE - Hessian: %e\n', MSE_h)
fprintf('MSE - Hessian (with baseline): %e\n', MSE_hbase)
fprintf('\n\n')

%% Gradient
fprintf('======= STD - GRADIENT ======= \n')
disp(num2str(std(G,1,2)));
fprintf('======= STD - GRADIENT WITH BASELINE ======= \n')
disp(num2str(std(Gbase,1,2)));

err = abs(G_ex - mean(G,2)).^2;
MSE_g = sum(err(:)) / numel(G_ex);
err = abs(G_ex - mean(Gbase,2)).^2;
MSE_gbase = sum(err(:)) / numel(G_ex);

fprintf('\n')
fprintf('MSE - Gradient: %e\n', MSE_g)
fprintf('MSE - Gradient (with baseline): %e\n', MSE_gbase)
fprintf('\n\n')

%% Return
fprintf('======= STD - RETURN ======= \n')
disp(num2str(std(J_est,1)));

err = abs(J_ex - mean(J_est)).^2;
MSE_j = sum(err(:)) / numel(J_ex);

fprintf('\n')
fprintf('MSE - Return: %e\n', MSE_j)
fprintf('\n\n')
