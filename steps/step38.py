if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dzrkai import Variable
from dzrkai.functions import reshape

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)