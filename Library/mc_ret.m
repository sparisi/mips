function R = mc_ret(data, gamma)
% Computes Monte-Carlo estimates of the return from potentially off-policy data.
% R_t = sum_(h=t)^T gamma^(h-t)*r_h

r = [data.r];
t = [data.t];
t(end+1) = 1;
R = zeros(size(r));

for k = size(R,2) : -1 : 1
    if t(k+1) == 1 % Next state is a new episode init state
        R(k) = r(k);
    else
        R(k) = r(k) + gamma * R(k+1);
    end
end
