import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
def draw_N_C_Acc(accuracy, N, C, k, isfull, path):
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
    ax.set_zlabel('Accuracy')
    plt.title('Accuracy Contour Plot for different N and C for Full%s' % str(isfull))
    fig.savefig(path + '/' + 'draw_N_C_Acc_%s.png' % ('Full' if isfull else 'Partial'))

def draw_N_C_Reject(delay_rate, N, C, k, isfull, path):
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
    fig.savefig(path + '/' + 'draw_N_C_Reject_%s.png' % ('Full' if isfull else 'Partial'))

def draw_N_C_Reject_Contour(delay_rate, N, C, k, isfull, path):
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
               extent=[0, len(nlist), 0, len(clist)])

    numxTicks = min(5, max(N))

    plt.xticks(np.linspace(0, len(nlist), numxTicks), [int(f) for f in np.linspace(1, max(N), numxTicks)])
    plt.yticks(np.linspace(0, len(clist), 11), [int(f) for f in np.linspace(0, max(C), 11)])

    plt.title('Reject Rate Contour Plot for different N and C for Full:%s' %str(isfull))
    plt.xlabel('N')
    plt.ylabel('C')
    plt.colorbar()

    fig.savefig(path + '/' + 'draw_N_C_Reject_Contour_%s.png' % ('Full' if isfull else 'Partial'))

def draw_N_C_Acc_Contour(accuracy, N, C, k, isfull, path):
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
               extent=[0, len(nlist), 0, len(acclist)])
    numxTicks = min(5, max(N))

    plt.xticks(np.linspace(0, len(nlist), numxTicks), [int(f) for f in np.linspace(1, max(N), numxTicks)])
    plt.yticks(np.linspace(0, len(acclist), 11), [int(f) for f in np.linspace(0, max(C), 11)])

    plt.title('Accuracy Contour Plot for different N and C for Full:%s' %str(isfull))
    plt.xlabel('N')
    plt.ylabel('C')
    plt.colorbar()

    fig.savefig(path + '/' + 'draw_N_C_Acc_Contour_%s.png' % ('Full' if isfull else 'Partial'))

def draw_Reject_Acc(Accuracy, Delay_rate, N, k, isfull, labels, path):
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
    pylab.legend(loc='lower right')
    pylab.xlabel('Reject Rate')
    pylab.ylabel('Accuracy')
    pylab.xlim([0,90])
    pylab.ylim([miny-3,100+5])
    pylab.title('Performance Comparison on %s Symbols Using %i Clusters' % ('Full' if isfull else 'Partial',k))

    plt.savefig(path + '/' + 'draw_Reject_Acc_%s.png' % ('Full' if isfull else 'Partial'))

def draw_n_Acc(accuracy, c, k, isfull, delay_rate, path):
    fig = plt.figure()
    maxN = max(key[1] for key in accuracy.keys())

    x, y = [], []
    for n in range(1, maxN + 1):
        if (k, n, c, isfull) in accuracy.keys():
            x.append(n)
            y.append(accuracy[(k, n, c, isfull)])
            plt.annotate('N:%i,Acc:%i%%,Rej:%i%%' % (n, accuracy[(k, n, c, isfull)], delay_rate[(k, n, c, isfull)]),\
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

    plt.savefig(path + '/' + ('draw_n_Acc_%i_%s.png' % (c,'Full' if isfull else 'Partial')))

def draw_K_Delay_Acc(accuracy, delay_rate, K, C, n, isfull, path):
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

    plt.ylim([0,90])

    plt.xlim([min(K), max(K)])
    plt.xlim([min(K), max(K)])

    plt.title('Mean Accuracies in %s Shapes with topN = %i' %(('Full' if isfull else 'Partial'), n))
    ax.set_zlabel('Accuracy')

    plt.savefig(path + '/' + 'draw_K_Delay_Acc_%s.png' % ('Full' if isfull else 'Partial'))