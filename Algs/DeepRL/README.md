**Credit goes to [Simon Ramstedt](https://github.com/rmst).**

In networks and layers, data is stored as row vectors (states, actions and parameters `W`). 
Sizes are
- `N` : number of samples,
- `I` : number of input,
- `O` : number of output.

Layers are decoupled, i.e., the activation `y(x) = f(x*w + b)` is expressed by three layers 
```
y_lin(x) = x*w  
y_bias(x) = x+b  
y_activation(x) = f(x)  
```

To create a network you can either manually provide all the layers, e.g.
```
nn = Network([ ...
    Lin(2,15) ...
    Bias(15) ...
    Sig() ...
    Lin(15,1) ...
    Bias(1) ...
    ]);
```
or you can input the desired sizes and activation functions, e.g.
```
nn = Network( [2, 15, 1], {'ReLU'} );
```
Using the latter initialization, a linear layer (plus bias) is always added at the end.
If you want to bound the output, add then another layer with
```
nn.set_output('Tanh'); % Or use any other bounded activation 
```
The weights of the last linear and bias layers have very small random weights, such that the output of the network is close to zero.
