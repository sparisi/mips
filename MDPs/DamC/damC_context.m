function [ctx, ctx_range] = damC_context()

ctx_range = [80, 120; % min action release (S_MIN_REL)
             20, 80;  % inflow (DAM_INFLOW)
             1, 2;    % surface (S)
             40, 60;  % upstream threshold (H_FLO_U)
             0.1, 1;  % turbine efficiency (ETA)
             2, 6;    % hydroelectric demand (W_HYD)
             30, 80;  % water demand (W_IRR)
             10, 50]; % downstream threshold (Q_FLO_D )

ctx = diff(ctx_range')' .* rand(8,1) + ctx_range(:,1);

end