function out = mcint(f, in, lo, hi, useSimplex)
% MCINT Monte Carlo numerical integration for integrals of any size.
%
%    INPUT
%     - f          : handle to the (vectorial column) function to integrate
%     - in         : [N x D] matrix containing the sample points (N is the 
%                    number of samples, D is the number of variables)
%     - lo         : lower bounds of integration
%     - hi         : upper bounds of integration
%     - useSimplex : if the points were sampled from the simplex
% 
%    OUTPUT
%     - out        : (vector column) result

D = length(lo);
N = size(in,1);

arg_in = cell(D,1);
for i = 1 : D
    arg_in{i} = in(:,i)';
end

z = f(arg_in{:});
s = sum(z,2);

if useSimplex
    pp = zeros(D+1, D);
    for i = 1 : D
        p = lo';
        p(i) = hi(i);
        pp(i,:) = p;
    end
    pp(end,:) = lo';
    [~, v] = convhulln(pp);
else
    v = abs(prod(hi - lo));
end

out = v * s / N;

end
