import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
def draw_N_C_Acc(accuracy, N, C, k, isfull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nmesh, cmesh = np.meshgrid(N, C)
    accmesh = [[None for j in range(len(nmesh[0]))] for i in range(len(nmesh))]

    nlist, clist, acclist =[], [], []

    for n in N:
        for c in C:
            nlist.append(n)
            clist.append(c)
            acclist.append(accuracy[k, n, c, isfull])

    ax.plot_trisurf(nlist, clist, acclist, cmap=plt.cm.jet, linewidth=0.2)
    '''
    for i in range(len(nmesh)):
        for j in range(len(nmesh[0])):
            # accuracy[(k, n, c, isFull)]
            accmesh[i][j] = accuracy[(k, nmesh[i][j], cmesh[i][j], isfull)]

    ax.plot_surface(nmesh, cmesh, accmesh, alpha=0.8, color='w', rstride=1, cstride=1, cmap=plt.cm.hot,
                    linewidth=0, antialiased=False)
    '''
    plt.xlabel('N')
    plt.ylabel('C')
    ax.set_zlabel('Accurcy')
    plt.show(block=True)

def draw_N_C_Reject(delay_rate, N, C, k, isfull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nmesh, cmesh = np.meshgrid(N, C)
    accmesh = [[None for j in range(len(nmesh[0]))] for i in range(len(nmesh))]

    nlist, clist, rejlist =[], [], []

    for n in N:
        for c in C:
            nlist.append(n)
            clist.append(c)
            rejlist.append(delay_rate[k, n, c, isfull])

    ax.plot_trisurf(nlist, clist, rejlist, cmap=plt.cm.jet, linewidth=0.2)
    '''
    for i in range(len(nmesh)):
        for j in range(len(nmesh[0])):
            # accuracy[(k, n, c, isFull)]
            accmesh[i][j] = accuracy[(k, nmesh[i][j], cmesh[i][j], isfull)]

    ax.plot_surface(nmesh, cmesh, accmesh, alpha=0.8, color='w', rstride=1, cstride=1, cmap=plt.cm.hot,
                    linewidth=0, antialiased=False)
    '''
    plt.xlabel('N')
    plt.ylabel('C')
    ax.set_zlabel('Reject Rate(%)')
    plt.show(block=True)

def draw_N_C_Reject_Contour(delay_rate, N, C, k, isfull):
    fig = plt.figure()

    nmesh, cmesh = np.meshgrid(N, C)

    nlist, clist, rejlist =[], [], []
    for n in N:
        for c in C:
            nlist.append(n)
            clist.append(c)
            rejlist.append(delay_rate[k, n, c, isfull])

    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(min(nlist), max(nlist), 100), np.linspace(min(clist), max(clist), 100)
    xi, yi = np.meshgrid(xi, yi)

    rbf = scipy.interpolate.Rbf(nlist, clist, rejlist, function='linear')
    zi = rbf(xi, yi)

    plt.imshow(zi, vmin=min(rejlist), vmax=max(rejlist), origin='lower',
               extent=[min(nlist), max(nlist), min(clist), max(clist)])

    plt.imshow(zi,  origin='lower')

    plt.title('Reject Rate Contour Plot for different N and C for Full:%s' %str(isfull))
    plt.xlabel('N')
    plt.ylabel('C')
    plt.colorbar()
    plt.show()
    plt.show(block=True)

def draw_N_C_Acc_Contour(accuracy, N, C, k, isfull):
    fig = plt.figure()

    nmesh, cmesh = np.meshgrid(N, C)
    accmesh = [[None for j in range(len(nmesh[0]))] for i in range(len(nmesh))]

    nlist, clist, acclist =[], [], []
    for n in N:
        for c in C:
            nlist.append(n)
            clist.append(c)
            acclist.append(accuracy[k, n, c, isfull])

    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(min(nlist), max(nlist), 100), np.linspace(min(clist), max(clist), 100)
    xi, yi = np.meshgrid(xi, yi)

    rbf = scipy.interpolate.Rbf(nlist, clist, acclist, function='linear')
    zi = rbf(xi, yi)

    plt.imshow(zi, vmin=min(acclist), vmax=max(acclist), origin='lower',
               extent=[min(nlist), max(nlist), min(clist), max(clist)])

    plt.imshow(zi,  origin='lower')

    plt.title('Accuract Contour Plot for different N and C for Full%s' %str(isfull))
    plt.xlabel('N')
    plt.ylabel('C')
    plt.colorbar()
    plt.show()
    plt.show(block=True)

def draw_Reject_Acc(Accuracy, Delay_rate, N, k, isfull, labels):
    import pylab
    # the only free variable is C, so we will vary C to see reject vs accuracy rate
    fig = plt.figure()
    c_list = [[key[2] for key in accuracy.keys()] for accuracy in Accuracy]
    miny= 0
    color = ['r', 'g', 'b', 'y']
    markers = ['x','.','x','.']
    for i in range(len(Accuracy)):
        for n in N:
            x_reject, y_acc = [], []
            for c in c_list[i]:
                x_reject.append(Delay_rate[i][(k, n, c, isfull)])
                y_acc.append(Accuracy[i][(k, n, c, isfull)])

            pylab.scatter(x_reject, y_acc, alpha=1, s=100, color=color[i], marker=markers[n+i], label=labels[i] + ' N='+str(n))
            miny= min (min(y_acc),miny)
        # labels
    pylab.legend(loc='upper right')
    pylab.xlabel('Reject Rate')
    pylab.ylabel('Accuracy')
    pylab.xlim([0,90])
    pylab.ylim([miny-3,100+5])
    pylab.title('Performance Comparison on Full Symbols Using %i Clusters' % k)

    pylab.show(block=True)

def draw_n_Acc(accuracy, c, k, isfull, delay_rate=None):
    fig = plt.figure()
    maxN = max(key[1] for key in accuracy.keys())

    x, y = [], []
    for n in range(1, maxN + 1):
        if (k, n, c, isfull) in accuracy.keys():
            x.append(n)
            y.append(accuracy[(k, n, c, isfull)])
            plt.annotate('N:%i,Acc:%i%%,Rej:%i%%' % (n, accuracy[(k, n, c, isfull)],delay_rate[(k, n, c, isfull)]),\
                         (n + 0.1, accuracy[(k, n, c, isfull)] + 0.1))


    # horizontal lines
    plt.plot([1 - 1, maxN + 1], [60, 60], 'r--')
    plt.plot([1 - 1, maxN + 1], [80, 80], 'g--')

    # labels
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.title('C:%i k:%i Full:%s' % (c, k, str(isfull)))

    plt.ylim(0, 100 * 1.1)
    plt.xlim(1 - 1, maxN + 1)
    plt.scatter(x, y, alpha=1, s=100)
    plt.grid(True)

    plt.show(block=True)

    '''
        Not finished
    '''

def draw_K_Delay_Acc(accuracy, delay_rate, K, C, n, isfull):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    klist, rejlist, acclist = [],[],[]
    for k in K:
        for c in C:
            if (k, n, c, isfull) in accuracy.keys():
                klist.append(k)
                rejlist.append(delay_rate[(k, n, c, isfull)])
                acclist.append(accuracy[(k, n, c, isfull)])

    ax.plot_trisurf(klist, rejlist, acclist, cmap=plt.cm.jet, linewidth=0.2)

    plt.xlabel('Count of Clusters')
    plt.ylabel('Delay Decision Rate')
    plt.title('Mean Accuracies in %s Shapes with topN = %i' %(('Full' if isfull else 'Partial'), n))
    ax.set_zlabel('Accuracy')

    plt.show(block=True)