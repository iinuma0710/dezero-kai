import unittest
import numpy as np

class Variable():
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None
        # 出力元の関数を保持する属性
        self.creator = None

    # creator 属性の setter
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            # self.data と同じ形状で要素がすべて1の ndarray インスタンスを生成
            self.grad = np.ones_like(self.data)

        funcs = [self.creator] 
        while funcs:    # 処理する関数がなくなるまで繰り返し
            f = funcs.pop()             # 関数を取得
            x, y = f.input, f.output    # 関数の入出力を取得
            x.grad = f.backward(y.grad) # backward メソッドの呼び出し
        
            if x.creator is not None:
                funcs.append(x.creator) # 1つ前の関数をリストに追加


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function():
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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
    

def square(x):
    return Square()(x)
    

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    

def exp(x):
    return Exp()(x)


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps) 


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.random(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
    

if __name__ == "__main__":
    # 順伝播
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    # 逆伝播
    y.backward()

    print(x.grad)