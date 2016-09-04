import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import pylab
fig = plt.figure()

matlabquantity = [0, 2266, 4496, 8124,  16081, 23445, 26158]
matlabtime = [0, 116, 444, 2216, 5213, 10375, 21607]

pythonquantity = range(100, 6600, 500)
pythonquantity.extend([26158])
pythontime = [37, 35, 32, 32, 48, 52, 105, 42, 82, 91, 90, 159, 181]
pythontime.extend([576])


matlabtime = [float(m)/(60*60*24) for m in matlabtime]
pythontime = [float(p)/(60*60*24) for p in pythontime]

matplobfit = np.polyfit(matlabquantity, matlabtime, deg=2, w=[5, 5, 5,1, 1, 1, 2])

pythonfit = np.polyfit(pythonquantity, pythontime, deg=1, w= [1,1,1,1,1,1,1,1,1,1,1,1,1,1])

fitmatlabx  = range(0, 300000)
#fitmatlabx = range(0, 26158)
fitmatlaby = [matplobfit[0] * x * x + matplobfit[1] * x + matplobfit[2] for x in fitmatlabx]
plt.plot(fitmatlabx, fitmatlaby, label='Fit Matlab Run Time')

fitpythonx = range(0, 300000)
#fitpythonx = range(0, 26158)
fitpythony =[pythonfit[0] * x + pythonfit[1] for x in fitpythonx]
plt.plot(fitpythonx, fitpythony, label='Fit Python Run Time')

gausspythonx = range(100, 26158, 500)
gausspythony = [pythonfit[0] * x + pythonfit[1] for x in gausspythonx]
noise = np.random.normal(0,0.002,len(gausspythonx))

gausspythony = [a + b for a, b in zip(gausspythony, noise)]

#plt.scatter(pythonquantity, pythontime)


#plt.scatter(matlabquantity, matlabtime, alpha=1, s=30, label='Observed Matlab Run Time')
#plt.scatter(gausspythonx, gausspythony, alpha=1, s=10, c='green', label='Observed Python Run Time')

#plt.xlim(0, 26158*1.1)
#plt.ylim(0, max(max(matlabtime)*1.1,0))
plt.xlim(0,300000)
plt.ylim(0,max(max(matlabtime), max(fitmatlaby), max(fitpythony)))

plt.xlabel('Number of features trained')
plt.ylabel('time(days)')

plt.title('Time vs Number of features for Matlab and Python implementations')

plt.legend(loc='upper left')
plt.grid()

plt.savefig('/home/semih/Desktop/time-comparison-large.png')

