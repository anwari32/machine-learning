#! usr/bin/python

# not working!!!

import numpy as np

class Linear_Regression_Model:
    def __init__(self, n_feature, learning_rate=0.00001, bias=0):
        self.weight = np.zeros(n_feature)
        self.learning_rate = learning_rate
        self.bias = bias

    def __str__(self):
        desc = "# features {}".format(self.weight.shape[0])
        desc += '\n' + "Learning rate {}".format(self.learning_rate)
        desc += '\n' + "Bias {}".format(self.bias)
        desc += '\n' + "Weights {}".format(self.weight)
        return desc

    # simple linear regression.
    def forward(self, xs):
        # compute linear combination of w and x
        # length of w and x must be the same.
        assert self.weight.shape == xs.shape, "input shape not match with weights shape."
        y_hat = 0
        for (_w, _x) in zip(self.weight, xs):
            y_hat += (_w * _x)
        
        y_hat += self.bias
        return y_hat
    
    def forward_vectorized(self, xs):
        assert self.weight.shape == xs.shape, "input shape not match with weights shape."
        y_hats = np.dot(self.weight, xs)
        y_hats += self.bias
        return y_hats
    
    def squared_error(y_hat, y):
        error_val = (y_hat - y) ** 2
        error_val /= 2
        return error_val
    
    def mean_squared_error(y_hats, ys):
        assert y_hats.shape == ys.shape, "output prediction shape not match with ground truth shape."
        m = y_hats.shape[0]
        temp = y_hats - ys
        temp = temp ** 2
        temp = np.sum(temp)
        temp /= (2 * m)
        return temp
        
    def train(self, xss, ys, upper_error=0.0001, lower_error=-0.0001):
        # repeat until minimum.
        repeat = True
        num_iter = 0
        final_error_val = 0
        while repeat:
            num_iter += 1
            for xs in xss:
                y_hats = self.forward_vectorized(xs)
                error_val = self.mean_squared_error(y_hats, ys)
                if not(error_val <= upper_error and error_val >= lower_error):
                    # backpropagation.
                    d_ws = (y_hats - ys) * xs
                    d_bias = (y_hats - ys)
                    # simulataneous update.
                    self.weight = self.weight - self.learning_rate * d_ws
                    self.bias = self.bias - self.learning_rate * d_bias
                else:
                    final_error_val = error_val
                    repeat = False
            
        print("training finished. final error {}, # iteration {}".format(final_error_val, num_iter))
