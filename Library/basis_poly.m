function Phi = basis_poly(degree, dim, offset, state)
% BASIS_POLY Computes full polynomial features: phi(s) = s^0+s^1+s^2+...
% Since s is a vector, s^i denotes all the possible products of degree i
% between all elements of s, e.g., 
%
% s = (a, b, c)'
% s^3 = a^3 + b^3 + c^3 + a^2b + ab^2 + ac^2 + a^2c + b^2c + bc^2
%
%    INPUT
%     - degree : degree of the polynomial
%     - dim    : dimension of the state
%     - offset : 1 if you want to include the 0-degree component,
%                0 otherwise
%     - state  : (optional) the state to evaluate
%
%    OUTPUT
%     - Phi              : if a state is provided as input, the function 
%                          returns the feature vector representing it; 
%                          otherwise it returns the number of features
%
% =========================================================================
% EXAMPLE
% basis_poly(2,3,1,[3,5,6]') = [1, 3, 5, 6, 15, 18, 30]'

if nargin == 3
    Phi = nmultichoosek(dim+1,degree);
    if ~offset
        Phi = Phi - 1;
    end
else
    assert(size(state,1) == dim)
    assert(size(state,2) == 1);
    C = nmultichoosek([1; state],degree);
    Phi = prod(C,2);
    if ~offset
        Phi(1) = [];
    end
end

return
