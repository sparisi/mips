clear, close all

% Prepare dataset
rng(3)
dimX = 2; dimY = 1;
N = 1000;
X = myunifrnd(-2*ones(dimX,1), 2*ones(dimX,1), N);
% X = normalize_data(X')';
Xbounds = [min(X,[],2) max(X,[],2)];

% Choose objective function
f = @(x) rosenbrock(x);
f = @(x) rastrigin(x);
f = @(x) quadcostmulti(x,0.5*Xbounds);
T = f(X); % Target

% Create network
nn = Network( [dimX, 15, dimY], {'Tanh'} );
% nn = Network( [dimX, 15, 50, 15, dimY], {'Tanh', 'Tanh', 'Tanh'} );
% nn = Network( [dimX, 5, 25, 75, 15, dimY], {'Tanh', 'Tanh', 'Tanh', 'Tanh'} );

optim = RMSprop(length(nn.W));
optim = ADAM(length(nn.W));
optim.alpha = 0.025;

loss = @(y,t) meansquarederror(y,t);

batchsize = 32;

t = 1;


%% Learn
tic
while t < 10000
    
    mb = randperm(N,batchsize);
    
    Y = nn.forwardfull(X(:,mb)')';
    [~, dL] = loss(Y,T(:,mb));
    dW = nn.backward(dL' / batchsize);
    nn.update(optim.step(nn.W, dW));
%     nn.update(nn.W - 0.01*dW);
    t = t + 1;
    
    Y_eval = nn.forward(X')';
    L_eval = loss(Y_eval,T);
    updateplot('Loss', t, mean(L_eval), 1)
%     updateplot('Network Parameters', t, nn.W)
%     updateplot('Gradient Norm', t, norm(dW), 1)
    if t == 2, autolayout, end
    
end
toc

%% Plotting
ft = @(x,y) f([x;y]);
fn = @(x,y) nn.forward([x;y]');
figure
subplot(1,2,1), fsurf(ft,[Xbounds(1,:) Xbounds(2,:)]), title('Target function');
subplot(1,2,2), fsurf(fn,[Xbounds(1,:) Xbounds(2,:)]), title('Neural network approx.');
