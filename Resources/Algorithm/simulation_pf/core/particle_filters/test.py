import numpy as np

a = np.array([0,0,0,0,0])
b = np.array([1,1,1,1,1])
c = np.vstack((a,b))
print((c>1.2).all)
