import numpy as np

class Variable():
    def __init__(self, data):
        self.data = data
        self.grad = None
        # 出力元の関数を保持する属性
        self.creator = None

    # creator 属性の setter
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator    # 1. 関数の取得
        if f is not None:
            x = f.input     # 2. 関数の入力を取得
            x.grad = f.backward(self.grad)  # 3. 関数の backward メソッド呼び出し
            x.backward()    # 再帰的に自分より一つ前の変数の backward メソッド呼び出し


class Function():
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        # 自分自身を出力する変数の creator に設定
        output.set_creator(self)
        self.input = input
        # 出力の変数も保持
        self.output = output
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

    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    # 逆伝播
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)