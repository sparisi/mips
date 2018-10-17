function R = mc_ret(data, gamma)
% Computes Monte-Carlo estimates of the return from potentially off-policy data.
% R_t = sum_(h=0)^T gamma^(h)*r_{t+h+1}

r = [data.r];
R = zeros(size(r));
terminal = [data.endsim]; % Be sure that the last step of each episode has endsim=true!

R_next = 0;
for k = size(R,2) : -1 : 1
    R(k) = r(k) + gamma * R_next * (1-terminal(k));
    R_next = R(k);
end
