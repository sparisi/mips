function h = HessianRF(policy, data, gamma, robj)

dlp = policy.dlogPidtheta;
hlp = policy.hlogPidtheta;
h = zeros(hlp, hlp);

num_trials = max(size(data));
parfor trial = 1 : num_trials
    sumrew = 0;
    sumhlogPi = zeros(hlp,hlp);
    sumdlogPi = zeros(dlp,1);
    for step = 1 : max(size(data(trial).a))
        dlogpidtheta = policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumdlogPi = sumdlogPi + dlogpidtheta;
        hlogpidtheta = policy.hlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));
        sumhlogPi = sumhlogPi + hlogpidtheta;
        sumrew = sumrew + gamma^(step-1) * data(trial).r(robj,step);
    end
    h = h + sumrew * (sumdlogPi * sumdlogPi' + sumhlogPi);
end

h = h / num_trials;

end