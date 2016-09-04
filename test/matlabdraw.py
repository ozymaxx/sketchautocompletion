
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as sio

mat_contents = sio.loadmat('accresult_full__niclcon_matlab.mat')

fig = plt.figure(figsize=(12, 12))
C = np.linspace(0, 100, 11)
N = range(1,14)

numxTicks = min(10, max(N))
plt.xticks(np.arange(0, 11, 1), np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, max(N) + 1, 1), np.arange(1, max(N) + 1, 1))


accmesh = []
for nidx, n in enumerate(N):
    temp = []
    for cidx, c in enumerate(C):
        temp.extend([mat_contents['accresult'][nidx,cidx]])
    accmesh.append(temp)

plt.imshow(accmesh, interpolation='none', cmap='summer')


isfull = True
title = 'for Matlab Implementation Using Niclcon dataset'
plt.title('Accuracy for  %s %s' % (('Full sketches' if isfull else 'Partial sketches'), title))
plt.xlabel('C')
plt.ylabel('N')
# plt.grid(ls='solid')

for nidx,n in enumerate(N):
    for cidx,c in enumerate(C):
        plt.text(c / 10 - 0.2, n - 1, '%.2f' % accmesh[nidx][cidx], fontsize=10)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(cax=cax)

plt.clim(0,100)
# plt.show()
plt.show()

fig.savefig('.' + '/' + 'draw_N_C_Acc_Contour_%s_Matlab.png' % ('Full' if isfull else 'Partial'))