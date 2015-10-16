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
str = ['\gamma = ' num2str(LQR.g) ', k = ' num2str(K) ', \sigma = ' num2str(sqrt(Sigma))];

syms x u F symSigma symK


%% Plot V function
F = lqr_vfunction(A,B,Q,R,K,Sigma,x,g);

figure
ezplot(F,[-10,10]);

title(['V-function, ' str])
xlabel('x'); ylabel('V(x)');


%% Plot Q function
F = lqr_qfunction(A,B,Q,R,K,Sigma,x,u,g);

figure
ezmesh(F,[-10,10,-10,10]);

title(['Q-function, ' str])
xlabel('x'); ylabel('u'); zlabel('Q(x,u)')


%% Plot A function
F = lqr_advantage(A,B,Q,R,K,Sigma,x,u,g);

figure
ezmesh(F,[-10,10,-10,10]);

title(['A-function, ' str])
xlabel('x'); ylabel('u'); zlabel('A(x,u)')


%% Plot J function
F = lqr_return(A,B,Q,R,symK,symSigma,x0,g);

figure
subplot(1,2,1,'align'), ezmesh(F,[-1.1,-0.3,0,0.5]);
xlabel('k'), ylabel('\sigma'), zlabel('J(k,\sigma)');
title(['\gamma = ' num2str(LQR.g) ', x_0 = ' mat2str(LQR.x0)])

subplot(1,2,2,'align'), ezcontourf(F,[-1.1,-0.3,0,0.5]);
xlabel('k'), ylabel('\sigma'), zlabel('J(k,\sigma)');
title(['\gamma = ' num2str(LQR.g) ', x_0 = ' mat2str(LQR.x0)])

axis equal
