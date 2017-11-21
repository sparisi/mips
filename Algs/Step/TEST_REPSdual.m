clear all
reset(symengine)

n = 5;
d = 3;
syms eta epsilon maxA
R = sym('R',[1,n]);
V = sym('V',[1,n]);
VN = sym('VN',[1,n]);
theta = sym('theta',[d,1]);
Phi = sym('Phi',[d,n]);
PhiN = sym('PhiN',[d,n]);


%% DUAL ETA
A = R + VN - V;
weights = exp( ( A - maxA ) / eta ); % Numerical trick
sumWeights = sum(weights);
sumWeightsA = (A - maxA) * transpose(weights);
sumWeightsAA = (A - maxA).^2 * transpose(weights);

% Dual function
g = eta * epsilon + eta * log(sumWeights/n) + maxA;
% Gradient wrt eta
gd = epsilon + log(sumWeights/n) - sumWeightsA / (eta * sumWeights);
% Hessian
h = (sumWeightsAA * sumWeights - sumWeightsA^2) / (eta^3 * sumWeights^2);

GD = jacobian(g,eta);
H = hessian(g,eta);

simplify(GD-gd)
simplify(H-h)


%% DUAL THETA
PhiDiff = PhiN - Phi;
A = R + transpose(theta) * PhiDiff;
weights = exp( ( A - maxA ) / eta ); % Numerical trick
sumWeights = sum(weights);
sumPhiWeights = PhiDiff * transpose(weights);
sumPhiWeightsPhi = PhiDiff * diag(weights) * transpose(PhiDiff);

% Dual function
g = eta * epsilon + eta * log(sumWeights/n) + maxA;
% Gradient wrt theta
gd = sumPhiWeights / sumWeights;
% Hessian
h = ( sumPhiWeightsPhi * sumWeights - sumPhiWeights * transpose(sumPhiWeights)) / sumWeights^2 / eta;

GD = transpose(jacobian(g,theta));
H = hessian(g,theta);

simplify(GD-gd)
simplify(H-h)
