import numpy as np
a= np.array([ 1.,  1.,  1])
print(a.shape)

b= np.array([[ 2.,  2.,  2.,  3.,  3.,  3.],
       [ 3.,  3.,  3.,  3.,  3.,  3.],
       [ 3.,  3.,  3.,  3.,  3.,  3.],
       [ 3.,  3.,  3.,  3.,  3.,  3.]])
a[:]=np.resize(b, a.shape) 
#b[:]=np.resize(a, b.shape) 
#b=a
print(a)
print(a.shape)