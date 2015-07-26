%% Init
clear all
clc
reset(symengine)

domain = 'lqr';
mdp_vars = lqr_mdpvariables;
dim = mdp_vars.dim;
robj = 1;

[N, pol, ~, steps, gamma] = feval([domain '_settings']);

LQR   = lqr_init(dim);
g     = LQR.g;
Q     = LQR.Q;
R     = LQR.R;
x0    = LQR.x0;
Sigma = LQR.Sigma;

J = sym('J',[1,dim]);
K = sym('k',[dim,dim]);
% k1_2 = 0;
% k2_1 = 0;
% K = subs(K);

for i = 1 : dim
    P = (Q{i}+K*R{i}*K)*(eye(dim)-g*(eye(dim)+2*K+K^2))^-1; % Only when A = B = I
    J(i) = -transpose(x0)*P*x0 + (1/(1-g))*trace(Sigma*(R{i}+g*P));
end
J = J(robj);

D_theta_J = transpose(jacobian(J,K(:)));
H_theta_J = hessian(J,K(:));


%% Run
trials = 2;
episodes = 1000;

for i = 1 : trials
    
    [ds, J_est] = collect_samples(domain, episodes, steps, pol);
    
    hess_est = HessianRF(pol, ds, gamma, robj);
    hess_estbase = HessianRFbase(pol, ds, gamma, robj);
    hess_ex = double(subs(H_theta_J,K(:),pol.theta));
    fprintf('\n ******* HESSIAN / TRIAL: %d *******\n\n', i)
    fprintf('Estimated:')
    fprintfmat(hess_est,size(hess_est,1),size(hess_est,2))
    fprintf('Estimated (with baseline):')
    fprintfmat(hess_estbase,size(hess_estbase,1),size(hess_estbase,2))
    fprintf('Exact:')
    fprintfmat(hess_ex,size(hess_ex,1),size(hess_ex,2))
    H(:,:,i) = hess_est;
    Hbase(:,:,i) = hess_estbase;
    
    grad_est = GPOMDP(pol, ds, gamma, robj);
    grad_estbase = GPOMDPbase(pol, ds, gamma, robj);
    grad_ex = double(subs(D_theta_J,K(:),pol.theta));
    fprintf('\n ******* GRADIENT / TRIAL: %d *******\n\n', i)
    fprintf('Estimated:')
    fprintfmat(grad_est,size(grad_estbase,1),size(grad_estbase,2))
    fprintf('Estimated (with baseline):')
    fprintfmat(grad_estbase,size(grad_estbase,1),size(grad_estbase,2))
    fprintf('Exact:')
    fprintfmat(grad_ex,size(grad_ex,1),size(grad_ex,2))
    G(:,i) = grad_est;
    Gbase(:,i) = grad_estbase;
    
    J_ex = double(subs(J,K(:),pol.theta));
    fprintf('\n ******* RETURN / TRIAL: %d *******\n\n', i)
    fprintf('Estimated: %.3f \n', J_est(robj))
    fprintf('Exact: %.3f \n', J_ex)
    JR(i) = J_est(robj);
    
    fprintf('\n::::::::::::::::::::::::::::::::::::::::::::\n\n')
    
end

%%
fprintf('\n ======= STD - HESSIAN ======= \n')
str = num2str(std(H,1,3));
disp(str);

fprintf('\n ======= STD - HESSIAN WITH BASELINE ======= \n')
str = num2str(std(Hbase,1,3));
disp(str);

fprintf('\n ======= STD - GRADIENT ======= \n')
str = num2str(std(G,1,2));
disp(str);

fprintf('\n ======= STD - GRADIENT WITH BASELINE ======= \n')
str = num2str(std(Gbase,1,2));
disp(str);

fprintf('\n ======= STD - RETURN ======= \n')
str = num2str(std(JR,1));
disp(str);

%%%
err = abs(grad_ex - mean(G,2)).^2;
MSE_g = sum(err(:)) / numel(grad_ex);
err = abs(grad_ex - mean(Gbase,2)).^2;
MSE_gbase = sum(err(:)) / numel(grad_ex);

fprintf('\n')

fprintf('MSE - Gradient: %e\n', MSE_g)
fprintf('MSE - Gradient (with baseline): %e\n', MSE_gbase)

%%%
err = abs(hess_ex - mean(H,3)).^2;
MSE_h = sum(err(:)) / numel(hess_ex);
err = abs(hess_ex - mean(Hbase,3)).^2;
MSE_hbase = sum(err(:)) / numel(hess_ex);

fprintf('\n')

fprintf('MSE - Hessian: %e\n', MSE_h)
fprintf('MSE - Hessian (with baseline): %e\n', MSE_hbase)