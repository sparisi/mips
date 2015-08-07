function fn = normalizeFront(f)
% Normalizes a frontier in order to have all the objectives in [0,1].

utopia = max(f);
antiutopia = min(f);
fn = bsxfun(@times,bsxfun(@plus,f,-antiutopia),1./(utopia-antiutopia));

end