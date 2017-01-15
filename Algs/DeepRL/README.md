**Credit goes to [Simon Ramstedt](https://git.ias.informatik.tu-darmstadt.de/u/SimonR)**

In networks and layers, data is stored as row vectors (states, actions and parameters `W`).  
Sizes are
- `N` : number of samples
- `I` : number of input
- `O` : number of output

Layers are decoupled, i.e., the activation `y(x) = f(x*w + b)` is expressed by three layers 
`
y_lin(x) = x*w
y_bias(x) = x+b
y_activation(x) = f(x)
`