function loss = eval_loss(front, domain)
% Evaluates the loss of an approximate frontier wrt a reference one, using
% a weigthed scalarization of the objectives.
%
% Reference: A Castelletti, F Pianosi and M Restelli (2012)
% Tree-based Fitted Q-Iteration for Multi-Objective Markov Decision Problems

[reference_front, weights] = feval([domain '_moref'],0);

diff_J = (max(reference_front) - min(reference_front));

loss = 0;

for i = 1 : size(weights,1)
    w = weights(i,:)';
    front_w = front * w;
    reference_front_w = reference_front * w;
    loss = loss + (max(reference_front_w) - max(front_w)) / (diff_J * w);
end

loss = loss / size(weights,1);

end