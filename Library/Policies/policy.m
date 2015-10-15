classdef (Abstract) policy
% POLICY Generic class for policies.
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        dim_explore; % Size of the explorative parameters (e.g., variance)
        dim; % Size of the action drawn by the policy
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta; % Policy parameters
    end
    
    methods
        %%% Enable arrays equality, i.e., 
        %%% [pol1, pol2] == pol3 or [pol1, pol2] == [pol3, pol4]
        function areEq = eq(obj1, obj2)
            n1 = numel(obj1);
            n2 = numel(obj2);
            
            if n1 == n2
                areEq = isequal(obj1,obj2);
            elseif n1 == 1
                areEq = false(size(obj2));
                for i = 1 : n2
                    areEq(i) = isequal(obj2(i),obj1);
                end
            elseif n2 == 1
                areEq = false(size(obj1));
                for i = 1 : n1
                    areEq(i) = isequal(obj1(i),obj2);
                end
            else
                error('Matrix dimensions must agree.')
            end
        end
        
        %%% Update function for policy gradient
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
    end
    
end

