classdef Network < handle
% Standard feedforward neural network.
    
    properties
       L % Layers
       W % Parameters of all layers
   end
   
   methods
       
       function obj = Network(varargin)
           if nargin == 1
               obj.L = varargin{1};
               obj.W = cat(2,obj.L.W);
           elseif nargin == 2
               dims = varargin{1};
               activs = varargin{2};
               obj.L = [Lin(dims(1),dims(2)), Bias(dims(2))];
               for i = 2 : numel(dims) - 1
                   obj.L = [obj.L ...
                       feval(activs{i-1}), Lin(dims(i),dims(i+1)), Bias(dims(i+1))];
               end
               obj.L(end).W = rand(size(obj.L(end).W))/1000;
               obj.L(end-1).W(:) = rand(size(obj.L(end-1).W))/1000;
               obj.W = cat(2,obj.L.W);
           end
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