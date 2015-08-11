function P = permutationMatrix(n, m)
% http://math.stackexchange.com/questions/307299/kronecker-product-and-the-commutation-matrix

P = zeros(n*m, n*m);
for k = 1 : n * m
    pos = mod((k-1),m)*n + floor((k-1)/m)+1; 
    P(k,pos) = 1;
end
