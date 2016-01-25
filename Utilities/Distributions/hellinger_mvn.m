function dist = hellinger_mvn(p, q)
% HELLINGER_MVN Computes the Hellinger distance between two Multivariate
% Normal distributions P and Q.

s1 = p.Sigma;
s2 = q.Sigma;
m1 = p.mu;
m2 = q.mu;

dist = 1 - ( det(s1)^0.25 * det(s2)^0.25 ) / ( det(0.5*s1 + 0.5*s2) )^0.5 * ...
    exp( 0.1250 * (m1 - m2)' / (0.5*s1 + 0.5*s2) * (m1 - m2) );
