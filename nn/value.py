class Value:
    def __init__(self, data, _prev=set(), _op="", _label=""):
        self.data = data
        self.previous = _prev
        self.operator = _op
        self.label = _label
        self.grad = 0
    
    def __repr__(self):
        return f"{self.__class__}(data={self.data})"

    def __add__(self, other):
        return Value(self.data + other.data, _prev=(self, other), _op="add")
    
    def __mul__(self, other):
        return Value(self.data * other.data, _prev=(self, other), _op="mul")
    
    def __sub__(self, other):
        return Value(self.data * other.data, _prev=(self, other), _op="sub")
    


if __name__ == "__main__":
    # define y = ax + b
    a = Value(2)    # a
    x = Value(5)    # x
    b = Value(-5)   # b
    y = a * x + b
    assert(a.data == 2)
    assert(x.data == 5)
    assert(b.data == -5)
    assert(y.data == 5)
    print(y)
