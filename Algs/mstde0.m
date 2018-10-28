function [g, gd, h] = mstde0(omega, Phi, T)
% Mean squared TD(0) error for learning linear value functions.
% Targets are constant, i.e., E = (omega'*phi - T)^2.

V = omega'*Phi;
TD = V - T; % T are the targets (constant)
g = 0.5*mean(TD.^2);
gd = Phi*TD'/size(T,2);
h = Phi*Phi'/size(T,2);
end
