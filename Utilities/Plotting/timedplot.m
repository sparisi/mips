function tim = timedplot(hP, hF, dt)
% TIMEDPLOT Automatically updates plots every DT seconds by using a timer.
%
%    INPUT
%     - hP  : handle of the plots to update (must be "ez" plots)
%     - hF  : handle to the function of each plot
%     - dt  : (optional) time interval between updates
%
%    OUTPUT
%     - tim : timer updating the plots

if nargin < 3, dt = 3; end

if ~iscell(hP), hP = num2cell(hP); end
if ~iscell(hF), hF = num2cell(hF); end

for i = 1 : numel(hP)
    updatefunctions{i} = @(~,~)ezupdate(hP{i},hF{i});
end

tim = timer('Period', dt, 'ExecutionMode', 'fixedRate');
tim.TimerFcn = @(~,~) cellfun(@feval,updatefunctions);
start(tim)