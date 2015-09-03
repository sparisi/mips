% Generates ccode translating the symbolic equations of the LQR problem

%% Init
clear all
clc
reset(symengine)

param_vec = {'P1', 'P2', 'P3'};
loss_vec  = {'utopia', 'antiutopia', 'pareto', 'mix1', 'mix2', 'mix3'};

%% Code generation
for param_type = param_vec

    for loss_type = loss_vec
        
        reset(symengine)
        
        [J, theta, rho, t, D_theta_J, D2_theta_J, D_t_theta, L] = ...
            settings_lqr3( loss_type, param_type );
        
        prefix = ['LQR3_' param_type '_' loss_type '_'];
        
        X = transpose(D_t_theta)*transpose(D_theta_J)*D_theta_J*D_t_theta;
        detX = X(1,1)*X(2,2)-X(1,2)*X(2,1);
        V = sqrt(detX);
        Jr = L*V;
        D_jr = transpose(jacobian(Jr,rho));
        
        for i = 1 : size(rho,2)
            str = strjoin(['ccode/' prefix 'Djr' num2str(i) '.ccode'],'');
            ccode(D_jr(i), 'file', str);
            disp(['Completed file ' str]);
        end
        
        str = strjoin(['ccode/' prefix 'Jr.ccode'], '');
        ccode(Jr, 'file', str);
        disp(['Completed file ' str]);
        
        clearvars -except param_type loss_type param_vec loss_vec
        
    end
    
end
