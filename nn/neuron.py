from value import Value

class Neuron:
    def __init__(self, inputs: Value, weights: Value, bias: Value):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.outputs = None


    def forward(self):
        self.outputs = self.inputs * self.weights + self.bias

    
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
