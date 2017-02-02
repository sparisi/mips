clear, clc, close all

rng(3)

train_in = load('mnist_small_train_in.txt');
train_labels = load('mnist_small_train_out.txt') + 1;
test_in = load('mnist_small_test_in.txt');
test_labels = load('mnist_small_test_out.txt') + 1;

n_labels = length(unique(train_labels));
train_n = size(train_in,1);
test_n = size(test_in,1);

train_out = zeros(train_n,n_labels);
train_out( sub2ind([train_n,n_labels], 1:train_n, train_labels') ) = 1;

test_out = zeros(test_n,n_labels);
test_out( sub2ind([test_n,n_labels], 1:test_n, test_labels') ) = 1;

dimX = size(train_in,2);
dimY = n_labels;
dimH1 = 600;

nn = Network([ ...
    Lin(dimX,dimH1) ...
    Bias(dimH1) ...
    ReLU() ...
    Lin(dimH1,dimY) ...
    Bias(dimY) ...
    Sig() ...
    ]);

optim = RMSprop(length(nn.W));
optim.alpha = 0.00025;

batchsize = 512;

t = 1;

%% Learn
while true
    
    mb = randperm(train_n,batchsize); 
    X = train_in(mb,:);
    T = train_out(mb,:);
    
    Y = nn.forwardfull(X);
    E = T - Y;
    L = mean(mean(E.^2));
    dL = - E / batchsize;
    dW = nn.backward(dL);
%     nn.update(optim.step(nn.W, dW)); % Use a gradient optimizer
    nn.update(nn.W - 0.2*dW); % No gradient optimizer
    t = t + 1;

    X_test = test_in;
    Y_test = nn.forward(X_test);
    [~, Y_test_labels] = max(Y_test,[],2);
    misclass_test = sum(test_labels ~= Y_test_labels) / length(Y_test_labels) * 100;
    
    updateplot('Loss', {t}, {L}, 1)
    updateplot('Misclassification Error (Percentage)', {t}, {misclass_test}, 1)
%     updateplot('Gradient Norm', {t}, {norm(dW)}, 1)
%     updateplot('Network Parameters', {t}, nn.W)
    if t == 2, autolayout, end
    
end