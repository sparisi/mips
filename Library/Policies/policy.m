classdef policy
    
    properties(GetAccess = 'public', SetAccess = 'protected')
        dim_explore; % Size of the explorative parameters (e.g., variance)
        dim; % Size of the action drawn by the policy
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta;
    end
    
    methods
        function areEq = eq(obj1, obj2)
            if ~strcmp(class(obj1),class(obj2))
                areEq = 0;
                return
            end
            if numel(obj1) ~= numel(obj2)
                if numel(obj1) == 1
                    areEq = false(size(obj2));
                    for i = 1 : numel(obj2)
                        areEq(i) = obj2(i) == obj1;
                    end
                elseif numel(obj2) == 1
                    areEq = false(size(obj1));
                    for i = 1 : numel(obj1)
                        areEq(i) = obj1(i) == obj2;
                    end
                else
                    error('Matrix dimensions must agree.')
                end
                return
            end
            areEq = sum(obj1.theta == obj2.theta) == size(obj1.theta,1);
        end
    end
    
end

