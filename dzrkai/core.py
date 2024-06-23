import weakref
import contextlib
import numpy as np

import dzrkai

try:
    import cupy
    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = (np.ndarray)


# =============================================================================
# Config
# =============================================================================

class Config:
    enable_backprop = True
    train = True


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


def test_mode():
    return using_config('train', False)


# =============================================================================
# Variable
# =============================================================================

class Variable():
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError("{} is not supported".format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p +')'

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dzrkai.functions.reshape(self, shape)
    
    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]

        return dzrkai.functions.transpose(self, axes)
    
    @property
    def T(self):
        return dzrkai.functions.transpose(self)
    
    def sum(self, axis=None, keepdims=False):
        return dzrkai.functions.sum(self, axis, keepdims)

    # creator 属性の setter
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def to_cpu(self):
        if self.data is not None:
            self.data = dzrkai.cuda.as_numpy(self.data)
            
    def to_gpu(self):
        if self.data is not None:
            self.data = dzrkai.cuda.as_cupy(self.data)

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = dzrkai.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

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

            with using_config('enable_backprop', create_graph):
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


class Parameter(Variable):
    pass


def as_array(x, array_modele=np):
    if np.isscalar(x):
        return array_modele.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


# =============================================================================
# Function
# =============================================================================

class Function():
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
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


class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dzrkai.functions.sum_to(gx0, self.x0_shape)
            gx1 = dzrkai.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:
            gx0 = dzrkai.functions.sum_to(gx0, self.x0_shape)
            gx1 = dzrkai.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if self.x0_shape != self.x1_shape:
            gx0 = dzrkai.functions.sum_to(gx0, self.x0_shape)
            gx1 = dzrkai.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 / x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if self.x0_shape != self.x1_shape:
            gx0 = dzrkai.functions.sum_to(gx0, self.x0_shape)
            gx1 = dzrkai.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = gy * c * x ** (c - 1)
        return gx


def neg(x):
    return Neg()(x)


def add(x0, x1):
    x1 = as_array(x1, dzrkai.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


def sub(x0, x1):
    x1 = as_array(x1, dzrkai.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, dzrkai.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


def mul(x0, x1):
    x1 = as_array(x1, dzrkai.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


def div(x0, x1):
    x1 = as_array(x1, dzrkai.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, dzrkai.cuda.get_array_module(x0.data))
    return Div()(x1, x0)
    

def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = dzrkai.functions.get_item