function copies = multicopy(obj, n)
% MULTICOPY Makes N copies of a copyable object OBJ.

assert(numel(obj) == 1, 'Can copy only one object at a time.')
copies = arrayfun(@(x)obj.copy, 1:n, 'UniformOutput', false);