clear all
close all

LQR = lqr_init(1);

A = LQR.A;
B = LQR.B;
Q = LQR.Q{1};
R = LQR.R{1};
g = LQR.g;
x0 = LQR.x0;
Sigma = LQR.Sigma;
K = -0.5;

syms x u F


%% Plot V function
F = lqr_vfunction(A,B,Q,R,K,Sigma,x,g);

figure
ezplot(F,[-10,10]);

legend(['\gamma = ' num2str(LQR.g) ', k = ' num2str(K)...
    ', \sigma = ' num2str(sqrt(Sigma))],'location','NorthOutside')
title('')
xlabel('x'); ylabel('V(x)');


%% Plot Q function
F = lqr_qfunction(A,B,Q,R,K,Sigma,x,u,g);

figure
ezmesh(F,[-10,10,-10,10]);

legend(['\gamma = ' num2str(LQR.g) ', k = ' num2str(K)...
    ', \sigma = ' num2str(sqrt(Sigma))],'location','NorthOutside')
title('')
xlabel('x'); ylabel('u'); zlabel('Q(x,u)')


%% Plot A function
F = lqr_advantage(A,B,Q,R,K,Sigma,x,u,g);

figure
ezmesh(F,[-10,10,-10,10]);

legend(['\gamma = ' num2str(LQR.g) ', k = ' num2str(K)...
    ', \sigma = ' num2str(sqrt(Sigma))],'location','NorthOutside')
title('')
xlabel('x'); ylabel('u'); zlabel('A(x,u)')


%% Plot J function
minK = -1.1;
maxK = -0.3;
stepK = 0.005;
minS = 0;
maxS = 0.5;
stepS = 0.005;

I = (maxK - minK) / stepK + 1;
J = (maxS - minS) / stepS + 1;
vK = zeros(I,1);
vS = zeros(1,J);
mJ = zeros(I,J);
J_opt = inf;
K_opt = inf;
S_opt = inf;
i = 1;

for K = minK : stepK : maxK
    vK(i) = K;
    j = 1;
    for Sigma = minS : stepS : maxS
        vS(j) = Sigma;
        J = lqr_return(A,B,Q,R,K,Sigma*Sigma,x0,g);
        mJ(i, j) = J;
        if J < J_opt
            J_opt = J;
            K_opt = K;
            S_opt = Sigma;
        end
        j = j + 1;
    end
    i = i + 1;
end

figure
subplot(1,2,1,'align'),mesh(vS,vK,mJ), xlabel('k'), ylabel('\sigma'), zlabel('J(k,\sigma)');
title(['\gamma = ' num2str(LQR.g) ', x_0 = ' mat2str(LQR.x0)])
subplot(1,2,2,'align'),contourf(vK,vS,mJ'), xlabel('k'), ylabel('\sigma'), zlabel('J(k,\sigma)');

axis equal
axis tight
