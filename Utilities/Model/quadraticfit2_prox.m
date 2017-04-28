function [model, nmse] = quadraticfit2_prox(X1, X2, Y, H, varargin)
% QUADRATICFIT2 Learns a quadratic model to fit input-output pairs 
% (X1, X2, Y)
% Y = X1'*R1*X1 + X2'*R2*X2 + 2*X1'*Rc*X2 + X1'*r1 + X2'*r2 + r0 
% (R1 is symmetric and negative definite). 
% The model is learned through proximal gradient.
%
% Equivalent formulation:
% Y = [ X1, X2, 1 ] H [ X1, X2, 1 ]'
% where
%     [ R1      Rc      0.5r1 ]
% H = [ Rc'     R2      0.5r2 ]
%     [ 0.5r1'  0.5r2'  r0    ]
%
%    INPUT
%     - X1          : [d1 x N] matrix, where N is the number of samples
%     - X2          : [d2 x N] matrix, where N is the number of samples
%     - Y           : [1 x N] vector
%     - H           : (optional) initial solution
%     - weights     : (optional) [1 x N] vector of samples weights
%     - standardize : (optional) flag to standardize X1, X2 and Y
%     - lambda_l1   : (optional) L1-norm regularizer
%     - lambda_l2   : (optional) L2-norm regularizer
%     - lambda_nn   : (optional) Nuclear norm regularizer
%     - lrate       : (optional) Proximal gradient learning rate
%     - maxiter     : (optional) Proximal gradient maximum number of 
%                     iterations
%
%    OUTPUT
%     - model       : struct with fields R1, R2, Rc, r1, r2, r0, dim 
%                     (number of parameters of the model) and eval 
%                     (function for generating samples, i.e., 
%                     Y = model.eval(X))
%     - nmse        : normalized mean squared error on the training data

%% Parse and preprocess input
[d1, N] = size(X1);
[d2, N] = size(X2);

p = inputParser;
p.KeepUnmatched = true;
addOptional(p, 'weights', ones(1,N));
addOptional(p, 'standardize', 0);
addOptional(p, 'lambda_l2', 0.001);
parse(p,varargin{:});

standardize = p.Results.standardize;
W = p.Results.weights;
lambda_l2 = p.Results.lambda_l2;

D = d1*(d1+1)/2 + d2*(d2+1)/2 + d1*d2 + d1 + d2 + 1; % Number of parameters of the quadratic model

if standardize
    [X1n, X1mu, X1std] = standardize_data(X1, W);
    [X2n, X2mu, X2std] = standardize_data(X2, W);
    [Yn, Ymu, Ystd] = standardize_data(Y, W);
else
    X1n = X1;
    X2n = X2;
    Yn = Y;
end

%% Find initial solution by linear regression
if isempty(H)
    
    % Generate features vector for linear regression
    Phi1 = basis_quadratic(d1,X1n); % Linear and quadratic features in X1
    Phi2 = basis_quadratic(d2,X2n); % Linear and quadratic features in X2
    Phi2(1,:) = []; % The bias is already in Phi1
    PhiC = bsxfun(@times, X1n, permute(X2n,[3 2 1])); % Cross features
    PhiC = permute(PhiC,[1 3 2]);
    PhiC = 2 * reshape(PhiC,[d1*d2,N]);
    
    Phi = [Phi1; Phi2; PhiC];
    
    % Get parameters by linear regression on pairs (Phi, Y) and extract model
    params = linear_regression(Phi, Yn, 'weights', W, 'lambda', lambda_l2);
    assert(~any(isnan(params)), 'Model fitting failed.')
    
    r0 = params(1,:);
    idx = 2;
    
    r1 = params(idx:d1+idx-1);
    idx = idx + d1;
    
    R1 = params(idx:idx+d1*(d1+1)/2-1);
    idx = idx + d1*(d1+1)/2;
    
    AQuadratic = zeros(d1);
    ind = logical(tril(ones(d1)));
    AQuadratic(ind) = R1;
    R1 = (AQuadratic + AQuadratic') / 2;
    
    r2 = params(idx:idx+d2-1);
    idx = idx + d2;
    
    R2 = params(idx:idx+d2*(d2+1)/2-1);
    idx = idx+d2*(d2+1)/2;
    
    AQuadratic = zeros(d2);
    ind = logical(tril(ones(d2)));
    AQuadratic(ind) = R2;
    R2 = (AQuadratic + AQuadratic') / 2;
    
    Rc = params(idx:end);
    Rc = reshape(Rc,d1,d2);
    
    H = [ [ [R1 Rc; Rc' R2] [0.5*r1; 0.5*r2] ] ; [ 0.5*r1' 0.5*r2' r0 ] ];

end

%% Perform APG
H = proxgrad(X1n, X2n, Yn, H, varargin{:});
R1 = H(1:d1, 1:d1);
R2 = H(d1+1:d1+d2, d1+1:d1+d2);
Rc = H(1:d1, d1+1:d1+d2);
r1 = 2 * H(1:d1, end);
r2 = 2 * H(d1+1:end-1, end);
r0 = H(end,end);

%% De-standardize
if standardize
    X1stdMat = diag((1./X1std));
    X2stdMat = diag((1./X2std));
    YstdMat = diag(Ystd);
    A1 = (X1stdMat * R1 * X1stdMat);
    A2 = (X2stdMat * R2 * X2stdMat);
    Ac = (X1stdMat * Rc * X2stdMat);
    R1 = YstdMat * A1;
    R2 = YstdMat * A2;
    Rc = YstdMat * Ac;
    newr1 = -YstdMat * (2*A1*X1mu + 2*Ac*X2mu - X1stdMat*r1);
    newr2 = -YstdMat * (2*A2*X2mu + 2*Ac'*X1mu - X2stdMat*r2);
    r0 = YstdMat * (X1mu'*A1*X1mu - r1'*X1stdMat*X1mu + ...
        X2mu'*A2*X2mu - r2'*X2stdMat*X2mu + ...
        2 * X1mu'*Ac*X2mu + ...
        r0) + Ymu;
    r1 = newr1;
    r2 = newr2;
	H = [ [ [R1 Rc; Rc' R2] [0.5*r1; 0.5*r2] ] ; [ 0.5*r1' 0.5*r2' r0 ] ];
end

%% Save model
model.dim = D;
model.R1 = R1;
model.R2 = R2;
model.Rc = Rc;
model.r1 = r1;
model.r2 = r2;
model.r0 = r0;
model.H = H;
model.eval = @(X1,X2) sum( (X1'*R1)' .* X1, 1 ) + ...
    2 * sum( (X1'*Rc)' .* X2, 1 ) + ...
    sum( (X2'*R2)' .* X2, 1 ) + ...
    (X1'*r1)' + ...
    (X2'*r2)' + ...
    r0';

if nargout > 1
    nmse = mean( ( Y - model.eval(X1,X2) ).^2 ) / mean( Y.^2 );
end
