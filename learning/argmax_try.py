import numpy as np

arr = np.array(range(10)).reshape(5, 2)
print(np.argmax(arr, axis=1))
