%% Create symbolic GMM
clear all
reset(symengine)

dim = 2;
ngauss = 3;

x = sym('x', [dim,1]);
MU = sym('m', [ngauss*dim,1]);
SIGMA = sym('s', [dim*dim*ngauss,1]);
ALPHA = sym('a', [ngauss,1]);

mix_c = exp(ALPHA) / sum(exp(ALPHA));

gmm_sym = 0;
for i = 1 : ngauss
    idx = (i-1)*dim;
    M = MU(1+idx:idx+dim,1);
    idx = idx * dim;
    S = SIGMA(1+idx:idx+dim^2,1);
    S = reshape(S, dim, dim);
    gd{i} = (2*pi)^(-dim/2) * det(S)^(-0.5)*exp(-0.5*transpose(x-M)/(S)*(x-M));
    gmm_sym = gmm_sym + mix_c(i) * gd{i};
end

%% Compute symbolic log gradient
grad_m = transpose(jacobian(log(gmm_sym), MU));
grad_s = transpose(jacobian(log(gmm_sym), SIGMA));
if ngauss == 1
    grad_a = [];
else
    grad_a = transpose(jacobian(log(gmm_sym), ALPHA));
end

grad_full = [grad_m; grad_s; grad_a];

%% Create numerical GMM
n_gauss  = ngauss;
n_params = dim;
mu = zeros(n_gauss,n_params);
sigma = zeros(n_params,n_params,n_gauss);
for i = 1 : n_gauss
    mu(i,:) = randi(10,[dim,1]);
    r = randi(10, dim);
    sigma(:,:,i) = r'*r;
end
p = rand(n_gauss,1);
p = p / sum(p);

if ngauss == 1
    policy = GaussianConstantChol(dim, mu', sigma);
else
    policy = GmmGibbsConstant(mu,sigma,n_gauss,p);
end

%% Check density
point = policy.drawAction(1);
tmpmu = mu';
tmpmu = tmpmu(:);
V1 = double(subs(gmm_sym, [MU; SIGMA; ALPHA; x], [tmpmu; sigma(:); p(:); point]));
V2 = policy.evaluate(point);
assert(abs(V1-V2)<1e-6)

%% Check gradient
G1 = double(subs(grad_full, [MU; SIGMA; ALPHA; x], [tmpmu; sigma(:); p(:); point]));
G2 = policy.dlogPidtheta(point);
assert(max(abs(G1-G2))<1e-6)
