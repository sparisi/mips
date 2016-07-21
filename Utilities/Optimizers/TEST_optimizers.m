clc, close all

rng(2)
r = rand(2,1);

f = @(x) sin(0.5*x(1)^2 - 0.25* x(2)^2 + 3)* cos(2*x(1)+1 - exp(x(2)));
fd = @(x)[sin(x(1).^2.*(1.0./2.0)-x(2).^2.*(1.0./4.0)+3.0).*sin(x(1).*2.0-exp(x(2))+1.0).*-2.0+x(1).*cos(x(1).^2.*(1.0./2.0)-x(2).^2.*(1.0./4.0)+3.0).*cos(x(1).*2.0-exp(x(2))+1.0),sin(x(1).^2.*(1.0./2.0)-x(2).^2.*(1.0./4.0)+3.0).*exp(x(2)).*sin(x(1).*2.0-exp(x(2))+1.0)-x(2).*cos(x(1).^2.*(1.0./2.0)-x(2).^2.*(1.0./4.0)+3.0).*cos(x(1).*2.0-exp(x(2))+1.0).*(1.0./2.0)];

[x1,~,~,output1] = fminsearch(f,r);
t1 = output1.iterations;
f1 = f(x1);
[x2,~,~,output2] = fminunc(f,r);
t2 = output2.iterations;
f2 = f(x2);

[x3,t3] = adam(f,fd,r);
f3 = f(x3);
[x4,t4] = rmsprop(f,fd,r);
f4 = f(x4);

[x1 x2 x3 x4]
[t1 t2 t3 t4]
[f1 f2 f3 f4]

%% min in f(1,3) = 0;
clc, close all

r = rand(2,1);

f = @(x) (x(1) + 2*x(2) - 7).^2 + (2*x(1) + x(2) - 5).^2;
fd = @(x) [ 10*x(1) + 8*x(2) - 34; 8*x(1) + 10*x(2) - 38];


[x1,~,~,output1] = fminsearch(f,r);
t1 = output1.iterations;
f1 = f(x1);
[x2,~,~,output2] = fminunc(f,r);
t2 = output2.iterations;
f2 = f(x2);

[x3,t3] = adam(f,fd,r);
f3 = f(x3);
[x4,t4] = rmsprop(f,fd,r);
f4 = f(x4);

[x1 x2 x3 x4]
[t1 t2 t3 t4]
[f1 f2 f3 f4]
