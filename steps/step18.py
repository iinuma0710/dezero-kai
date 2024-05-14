import weakref
import contextlib
import numpy as np

class Variable():
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        # 世代管理をするための属性
        self.generation = 0

    # creator 属性の setter
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        # 逆伝播で通っていない関数を generation でソートしながら追加
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # 出力元の関数の全出力をリストにまとめる
            gys = [output().grad for output in f.outputs]
            # 出力元の関数の逆伝播を実行
            gxs = f.backward(*gys)
            # 戻り値がタプルでない場合の処理
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            # 微分を保持しない場合
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y は weakref


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
        
        # 世代情報の設定は逆伝播ありの場合に行う
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()
    

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    

def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data # <= self.input.data から変更
        gx = 2 * x * gy
        return gx
    

def square(x):
    return Square()(x)


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name, value)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)
    

if __name__ == "__main__":
    # x0 = Variable(np.array(1.0))
    # x1 = Variable(np.array(1.0))
    # t = add(x0, x1)
    # y = add(x0, t)
    # y.backward()
    # print(y.grad, t.grad)
    # print(x0.grad, x1.grad)

    x = Variable(np.ones((100, 100, 100)))
    y = square(square(square(x)))
    y.backward()

    with no_grad():
        x = Variable(np.ones((100, 100, 100)))
        y = square(square(square(x)))