import numpy as np
import Loader
GRADE = 3
A = np.arange(48)
#print(A)
A = A.reshape((4, 2, 2, 3))
#print(A)
B = A[0, :, :, 0].flatten()
#print(B)
bins = [0, 60, 80]
Loader.histogram(A, bins, 2)
C = np.arange(12)
C = C.reshape((4,3))
print(C)
C = Loader.polynom(C,3)
print(C)