import numpy as np
import glob
import os

file = "/home/gor4i4ka/tmp/test"
file1 = "/home/gor4i4ka/tmp/test.npy"
file2 = open("/home/gor4i4ka/tmp/test.txt", "a+")


dummy = np.zeros((2, 3))
dummy1 = np.copy(dummy)
dummy2 = np.copy(dummy)
for i in range(dummy.shape[0]):
    for j in range(dummy.shape[1]):
        dummy[i][j] = i*3 + j
print(dummy)
dummy.tofile(file)

for i in range(dummy1.shape[0]):
    for j in range(dummy1.shape[1]):
        dummy1[i][j] = 9

print(dummy1)

dummy1.tofile(file)

dummy2 = np.fromfile(file)

print(dummy2, "TTTTTT")

np.save(file2, dummy)
#file2.close()
#file3 = open("/home/gor4i4ka/tmp/test.txt", "a")
np.savetxt(file2, dummy1)
#file3.close()
#file4 = open("/home/gor4i4ka/tmp/test.txt", "r")
dummy2 = np.load(file2)
print(dummy2)


