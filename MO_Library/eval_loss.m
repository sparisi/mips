function loss = eval_loss(front, mdp)
% EVAL_LOSS Evaluates the loss of an approximate frontier wrt a reference 
% one, using a weigthed scalarization of the objectives.
%
% =========================================================================
% REFERENCE
% A Castelletti, F Pianosi and M Restelli
% Tree-based Fitted Q-Iteration for Multi-Objective Markov Decision 
% Problems (2012)

[reference_front, weights] = mdp.truefront;
diff_max = (max(reference_front) - min(reference_front));


loss = 0;
for i = 1 : size(weights,1)
    w = weights(i,:)';
    front_w = front * w;
    reference_front_w = reference_front * w;
    loss_i = (max(reference_front_w) - max(front_w)) / (diff_max * w);
    if loss_i > 0, warning('Positive loss!'), end
    loss = loss + loss_i;
end
loss = loss / size(weights,1);


% fw = front*weights';
% rw = reference_front*weights';
% loss = sum( ...
%     ( max(rw,[],1) - max(fw,[],1) ) ./ ( diff_max * weights') ...
%     ) / size(weights,1);
