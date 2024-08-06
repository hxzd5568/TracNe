import numpy as np

a = np.array([1.0, 1.0, 1])
print(a.shape)

b = np.array(
    [
        [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    ]
)
a[:] = np.resize(b, a.shape)
# b[:]=np.resize(a, b.shape)
# b=a
print(a)
print(a.shape)
