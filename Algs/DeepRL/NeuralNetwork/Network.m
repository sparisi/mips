classdef Network < handle
% Standard feedforward neural network.
    
    properties
       L % Layers
       W % Parameters of all layers
   end
   
   methods
       
       function obj = Network(L)
           obj.L = L;
           obj.W = cat(2,obj.L.W);
       end
       
       function update(obj, W)
           if ~isrow(W), W = W'; end
           assert(isrow(W), 'W must be a vector, not a matrix.')
           assert(length(W) == length(obj.W), 'Wrong length.')
           obj.W = W;
       end
       
       function cpObj = copy(obj)
           cpObj = Network(obj.L.copy);
           cpObj.W = obj.W;
       end
       
       function X = forward(obj, X)
           idx = 1;
           for l = obj.L
               dimW = numel(l.W);
               l.W = obj.W(idx : idx + dimW - 1);
               X = l.forward(X);
               idx = idx + dimW;
           end
       end
       
       function X = forwardfull(obj, X)
           idx = 1;
           for l = obj.L
               dimW = length(l.W);
               l.W = obj.W(idx : idx + dimW - 1);
               X = l.forwardfull(X);
               idx = idx + dimW;
           end
       end
       
       function [dW, dE] = backward(obj, dE)
           idx = length(obj.W);
           dW = zeros(1,idx);
           for l = flip(obj.L)
               dimW = length(l.W);
               l.W = obj.W(idx - dimW + 1 : idx);
               [dE, dWl] = l.backward(dE);
               dW(idx - dimW + 1 : idx) = dWl;
               idx = idx - dimW;
           end
       end
       
   end
   
end