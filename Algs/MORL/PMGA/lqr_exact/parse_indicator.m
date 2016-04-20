function [I, I_params] = parse_indicator(indicator_type, D_theta_J, J, Up, AUp)
% PARSE_INDICATOR Return the symbolic equation of the desired indicator.

I_params = {};
if strcmp('pareto',indicator_type)
    I = -paretoDirectionNorm(D_theta_J);
elseif strcmp('utopia',indicator_type)
    I = -norm(J./Up-ones(size(J)))^2;
elseif strcmp('antiutopia',indicator_type)
    I = norm(J./AUp-ones(size(J)))^2;
elseif strcmp('mix1',indicator_type)
    I_params = sym('beta');
    I_antiutopia = norm(J./AUp-ones(size(J)))^2;
    I_pa = paretoDirectionNorm(D_theta_J);
    I = I_antiutopia*(1-I_params*I_pa);
elseif strcmp('mix2',indicator_type)
    I_params = sym('beta',[1,2]);
    I_utopia = norm(J./Up-ones(size(J)))^2;
    I_antiutopia = norm(J./AUp-ones(size(J)))^2;
    I = I_params(1) * I_antiutopia / I_utopia - I_params(2);
elseif strcmp('mix3',indicator_type)
    I_params = sym('beta');
    I_antiutopia = norm(J./AUp-ones(size(J)))^2;
    I_utopia = norm(J./Up-ones(size(J)))^2;
    I = I_antiutopia*(1-I_params*I_utopia);
else
    error('Unknown indicator.')
end
