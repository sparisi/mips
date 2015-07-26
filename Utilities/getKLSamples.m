function div = getKLSamples(p, q, states, actions)
% Approximates the Kullback-Leibler KL(q||p) divergence beetween two
% distributions q (the old one) and p (the new one) using samples.
% Reference: http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

n = size(actions,1);
pWeighting = zeros(n,1);
qWeighting = zeros(n,1);

parfor i = 1 : n
    pWeighting(i) = p.evaluate(states(i,:)', actions(i,:)');
    qWeighting(i) = q.evaluate(states(i,:)', actions(i,:)');
end

qWeighting = qWeighting / sum(qWeighting);
pWeighting = pWeighting / sum(pWeighting);
index = pWeighting > 10^-10;
qWeighting = qWeighting(index);
pWeighting = pWeighting(index);

div = sum(pWeighting .* log(pWeighting ./ qWeighting));

end
