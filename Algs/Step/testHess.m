%% Init
clear all
clc
reset(symengine)

domain = 'lqr';
mdp_vars = lqr_mdpvariables;
dim = mdp_vars.dim;
robj = 1;

[N, pol, ~, ~, gamma] = feval([domain '_settings']);

LQR   = lqr_init(dim);
g     = LQR.g;
Q     = LQR.Q;
R     = LQR.R;
x0    = LQR.x0;
Sigma = LQR.Sigma;

J = sym('J',[1,dim]);
K = sym('k',[dim,dim]);
k1_2 = 0; k2_1 = 0; K = subs(K);
theta = nonzeros(K);
dtheta = length(theta);

for i = 1 : dim
    P = (Q{i}+K*R{i}*K)*(eye(dim)-g*(eye(dim)+2*K+K^2))^-1; % Only when A = B = I
    J(i) = -transpose(x0)*P*x0 + (1/(1-g))*trace(Sigma*(R{i}+g*P));
end
J = J(robj);

D_theta_J = transpose(jacobian(J,theta));
H_theta_J = hessian(J,theta);


%% Run
trials = 2;
episodes = 200;
steps = 50;

Hex = double(subs(H_theta_J,theta,pol.theta));
Gex = double(subs(D_theta_J,theta,pol.theta));
Jex = double(subs(J,theta,pol.theta));

for i = 1 : trials
    
    [ds, J_est, G_est, H_est] = collect_samples_rele(domain, episodes, steps, pol);
    
    %% Hessian
    Hrele(:,:,i) = H_est{robj};
    H(:,:,i) = HessianRF(pol, ds, gamma, robj);
    Hbase(:,:,i) = HessianRFbase(pol, ds, gamma, robj);

    fprintf('\n ******* HESSIAN / TRIAL: %d *******\n\n', i)
    fprintf('Estimated:')
    fprintfmat(H(:,:,i),dtheta,dtheta)
    fprintf('Estimated (with baseline):')
    fprintfmat(Hbase(:,:,i),dtheta,dtheta)
    fprintf('Estimated (ReLe):')
    fprintfmat(Hrele(:,:,i),dtheta,dtheta)
    fprintf('Exact:')
    fprintfmat(Hex,dtheta,dtheta)

%     %% Gradient
%     Grele(:,i) = G_est(:,robj);
%     G(:,i) = GPOMDP(pol, ds, gamma, robj);
%     Gbase(:,i) = GPOMDPbase(pol, ds, gamma, robj);
%     
%     fprintf('\n ******* GRADIENT / TRIAL: %d *******\n\n', i)
%     fprintf('Estimated:')
%     fprintfmat(G(:,i),dtheta,1)
%     fprintf('Estimated (with baseline):')
%     fprintfmat(Gbase(:,i),dtheta,1)
%     fprintf('Estimated (ReLe):')
%     fprintfmat(Grele(:,i),dtheta,1)
%     fprintf('Exact:')
%     fprintfmat(Gex,dtheta,1)
%     
%     %% Return
%     JR(i) = J_est(robj);
%     fprintf('\n ******* RETURN / TRIAL: %d *******\n\n', i)
%     fprintf('Estimated: %.3f \n', J_est(robj))
%     fprintf('Exact: %.3f \n', Jex)
%     
%     fprintf('\n::::::::::::::::::::::::::::::::::::::::::::\n\n')
    
end

%% Print
%% Hessian
fprintf('\n ======= STD - HESSIAN ======= \n')
disp(num2str(std(H,1,3)));
fprintf('\n ======= STD - HESSIAN WITH BASELINE ======= \n')
disp(num2str(std(Hbase,1,3)));
fprintf('\n ======= STD - HESSIAN RELE ======= \n')
disp(num2str(std(Hrele,1,3)));

err = abs(Hex - mean(H,3)).^2;
MSE_h = sum(err(:)) / numel(Hex);
err = abs(Hex - mean(Hbase,3)).^2;
MSE_hbase = sum(err(:)) / numel(Hex);
err = abs(Hex - mean(Hrele,3)).^2;
MSE_hrele = sum(err(:)) / numel(Hex);

fprintf('\n')
fprintf('MSE - Hessian: %e\n', MSE_h)
fprintf('MSE - Hessian (with baseline): %e\n', MSE_hbase)
fprintf('MSE - Hessian (ReLe): %e\n', MSE_hrele)
fprintf('\n\n')

% %% Gradient
% fprintf('\n ======= STD - GRADIENT ======= \n')
% disp(num2str(std(G,1,2)));
% fprintf('\n ======= STD - GRADIENT WITH BASELINE ======= \n')
% disp(num2str(std(Gbase,1,2)));
% fprintf('\n ======= STD - GRADIENT RELE ======= \n')
% disp(num2str(std(Grele,1,2)));
% 
% err = abs(Gex - mean(G,2)).^2;
% MSE_g = sum(err(:)) / numel(Gex);
% err = abs(Gex - mean(Gbase,2)).^2;
% MSE_gbase = sum(err(:)) / numel(Gex);
% err = abs(Gex - mean(Grele,2)).^2;
% MSE_grele = sum(err(:)) / numel(Gex);
% 
% fprintf('\n')
% fprintf('MSE - Gradient: %e\n', MSE_g)
% fprintf('MSE - Gradient (with baseline): %e\n', MSE_gbase)
% fprintf('MSE - Gradient (ReLe): %e\n', MSE_grele)
% fprintf('\n\n')
% 
% %% Return
% fprintf('\n ======= STD - RETURN ======= \n')
% disp(num2str(std(JR,1)));
% 
% err = abs(Jex - mean(JR)).^2;
% MSE_j = sum(err(:)) / numel(Jex);
% 
% fprintf('\n')
% fprintf('MSE - Return: %e\n', MSE_j)
% fprintf('\n\n')
