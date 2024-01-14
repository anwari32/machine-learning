from math import exp

def tanh(x):
    v = (exp(2 * x) - 1) / (exp(2 * x) + 1)
    return v

def dtanh(x):
    v = 1 - (tanh(x)**2)
    return v


if __name__ == "__main__":
    a = 0.8814
    y = dtanh(a)
    assert(round(y, 5) == 0.49998)
