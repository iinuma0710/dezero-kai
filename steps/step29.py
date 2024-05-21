if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dzrkai import Function, Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

def gx2(x):
    return 12 * x ** 2 -4

if __name__ == "__main__":
    x = Variable(np.array(2.0))

    iters = 10  # 繰り返し回数
    
    for i in range(iters):
        print(i, x)

        y = f(x)

        # 勾配の計算
        x.cleargrad()
        y.backward()

        # x の更新
        x.data -= x.grad / gx2(x.data)