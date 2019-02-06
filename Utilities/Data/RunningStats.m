classdef RunningStats < handle
    % https://github.com/openai/baselines/blob/d34049cab46908614c46aba1a201bf772daffeb0/baselines/common/running_mean_std.py
    
    properties
        mean
        var
        count
    end
    
    methods
        function obj = RunningStats(dim)
            obj.mean = zeros(dim,1);
            obj.var = ones(dim,1);
            obj.count = 0;
        end
        
        function update(obj, x)
            batch_mean = mean(x, 2);
            batch_var = var(x, [], 2);
            batch_count = size(x, 2);
            
            delta = batch_mean - obj.mean;
            tot_count = obj.count + batch_count;
            
            new_mean = obj.mean + delta * batch_count / tot_count;
            m_a = obj.var * obj.count;
            m_b = batch_var * batch_count;
            M2 = m_a + m_b + delta.^2 * obj.count * batch_count / tot_count;
            new_var = M2 / tot_count;
            new_count = tot_count;
            
            obj.count = new_count;
            obj.mean = new_mean;
            obj.var = new_var;
        end
        
        
        function x = standardize(obj, x)
            m = obj.mean;
            s = sqrt(obj.var);
            s(s==0) = 1.;
            x = bsxfun(@times, bsxfun(@minus, x, m), 1./s);
        end
    end
end