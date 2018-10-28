function [g, gd, h] = mstde_v(omega, Phi, T)
% Mean squared TD error for learning V.
V = omega'*Phi;
TD = V - T; % T are the targets (constant)
g = 0.5*mean(TD.^2);
gd = Phi*TD'/size(T,2);
h = Phi*Phi'/size(T,2);
end
