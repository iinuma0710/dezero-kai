import numpy as np

class Variable():
    def __init__(self, data):
        self.data = data
        # 微分値を保持する属性
        self.grad = None


class Function():
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        # 入力された変数を保持する
        self.input = input
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
    

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    # 順伝播
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 逆伝播
    y.grad = 1
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)