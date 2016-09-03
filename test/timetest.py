import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import pylab
fig = plt.figure()

matlabquantity = [0, 2266, 4496, 16081, 23445, 26158]
matlabtime = [0, 116, 444, 3213, 10375, 21607]

pythonquantity = range(100, 6600, 500)
pythonquantity.extend([26158])
pythontime = [37, 35, 32, 32, 48, 52, 105, 42, 82, 91, 90, 159, 181]
pythontime.extend([576])

plt.scatter(pythonquantity, pythontime)

plt.scatter(matlabquantity, matlabtime, alpha=1, s=100)
#plt.plot(matlabquantity, matlabtime)

matplobfit = np.polyfit(matlabquantity, matlabtime, deg=2, w=[5, 5, 5, 1, 1, 2])

pythonfit = np.polyfit(pythonquantity, pythontime, deg=1, w= [1,1,1,1,1,1,1,1,1,1,1,1,1,1])

#fitmatlabx  = range(0, 300000)
fitmatlabx = range(0, 26158)
fitmatlaby = [matplobfit[0] * x * x + matplobfit[1] * x + matplobfit[2] for x in fitmatlabx]
plt.plot(fitmatlabx, fitmatlaby)


#fitpythonx = range(0, 300000)
fitpythonx = range(0, 26158)
fitpythony =[pythonfit[0] * x + pythonfit[1] for x in fitpythonx]
plt.plot(fitpythonx, fitpythony)


plt.xlim(0, 26158+100)
plt.ylim(0, max(max(matlabtime),1)+999)
#plt.xlim(0,300000)
#plt.ylim(0,max(max(matlabtime), max(fitmatlaby), max(fitpythony)))

plt.xlabel('Number of features trained')
plt.ylabel('time(in seconds)')

plt.title('Time vs Number of features for Matlab and Python implementations')
plt.grid()
print fitmatlaby[len(fitmatlaby) - 1] / (60 * 60 * 24)
plt.show()
