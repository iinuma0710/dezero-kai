import numpy as np

# CuPy がインストールされていない場合のエラー回避
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False

from dzrkai import Variable


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data

    # GPU が無効の場合は常に numpy のモジュールを返す
    if not gpu_enable:
        return np
    
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    
    return cp.asnumpy(x)


def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception('CuPy cannot be loaded. Install CuPy!')
    
    return cp.asarray(x)
