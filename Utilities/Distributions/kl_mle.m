function div = kl_mle(pWeighting, qWeighting)
% KL_MLE Approximates the Kullback-Leibler KL(Q||P) divergence beetween two
% distributions Q (the old one) and P (the new one) using the weights for
% a Maximum-Likelihood update.
% If no weights for Q are provided, they are assumed to be 1.

if(nargin == 1), qWeighting = ones(1,length(pWeighting)); end

qWeighting = qWeighting / sum(qWeighting);
pWeighting = pWeighting / sum(pWeighting);
index = pWeighting > 10^-10;
qWeighting = qWeighting(index);
pWeighting = pWeighting(index);

div = sum(pWeighting .* log(pWeighting ./ qWeighting));
