% Generates ccode translating the symbolic equations of the LQR problem

%% Init
clear all
clc
reset(symengine)

script_path = mfilename('fullpath');
script_path = script_path(1:end-length(mfilename));

param_vec = {'P1', 'P2'};
indicator_vec = {'utopia', 'antiutopia', 'pareto', 'mix1', 'mix2', 'mix3'};

%% Code generation
for param_type = param_vec

    for indicator_type = indicator_vec
        
        reset(symengine)
        
        [J, theta, rho, t, D_theta_J, D2_theta_J, D_t_theta, ~, I] = ...
            settings_lqr3( indicator_type, param_type );
        
        prefix = ['LQR3_' param_type '_' indicator_type '_'];
        
        X = transpose(D_t_theta)*transpose(D_theta_J)*D_theta_J*D_t_theta;
        detX = X(1,1)*X(2,2)-X(1,2)*X(2,1);
        V = sqrt(detX);
        L = I*V;
        D_L = transpose(jacobian(L,rho));
        
        for i = 1 : length(rho)
            str = strjoin([script_path 'ccode/' prefix 'DL' num2str(i) '.ccode'],'');
            ccode(D_L(i), 'file', str);
            disp(['Completed file ' str]);
        end
        
        str = strjoin([script_path 'ccode/' prefix 'L.ccode'], '');
        ccode(L, 'file', str);
        disp(['Completed file ' str]);
        
        clearvars -except param_type indicator_type param_vec indicator_vec script_path
        
    end
    
end
