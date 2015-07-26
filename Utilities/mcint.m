function out = mcint(f, in, lo, hi, useSimplex)
% Monte Carlo numerical integration for integrals of any size.
%
% Inputs:
%  - f          : handle to the (vectorial column) function to integrate
%  - in         : N-by-dim matrix containing the sample points (N is the 
%                 number of samples, dim is the number of variables)
%  - lo         : lower bounds of integration
%  - hi         : upper bounds of integration
%  - useSimplex : if the points were sampled from the simplex
% 
% Outputs:
%  - out      : (vector column) result

dim = length(lo);
N = size(in,1);

arg_in = cell(dim,1);
for i = 1 : dim
    arg_in{i} = in(:,i)';
end

z = f(arg_in{:});
s = sum(z,2);

if useSimplex
    pp = zeros(dim+1, dim);
    for i = 1 : dim
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
