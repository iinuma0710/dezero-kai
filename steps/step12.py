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
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)          # 引数をアンパッキングして渡す
        if not isinstance(ys, tuple):   # 戻り値がタプルでない場合の対応
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    

def add(x0, x1):
    return Add()(x0, x1)
    

if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data)