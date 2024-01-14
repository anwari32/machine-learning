from nn.value import Value
from nn.activation_functions import tanh, dtanh


class Neuron:
    def __init__(self, inputs: float, weights: Value, bias: Value, activation=tanh):
        self.inputs: Value = inputs
        self.weights: Value = weights
        self.bias: Value = bias
        self.activation = activation
        self.outputs: Value = None

    def forward(self):
        self.outputs = self.inputs * self.weights + self.bias
        self.outputs = self.activation(self.outputs)



    
if __name__ == "__main__":
    inputs = Value(2)
    weights = Value(3)
    bias = Value(5)
    n = Neuron(inputs, weights, bias)
    n.forward()
    assert(n.inputs.data == 2)
    assert(n.weights.data == 3)
    assert(n.bias.data == 5)
    print(n.outputs)
