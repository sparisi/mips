function y = rosenbrock_ctx(x,p,G)
% This version of the Rosenbrock function perturbates the parameters X
% according to X = X + G'*P, where G is a fixed matrix.

x = x + G'*p;
y = -sum( 100*(x(2:end,:) - x(1:end-1,:).^2).^2 + (1 - x(1:end-1,:)).^2 , 1 );
