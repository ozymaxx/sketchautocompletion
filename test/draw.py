"""
Methods for plotting accuracy and reject rate results
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import pylab
from mpl_toolkits.axes_grid1 import make_axes_locatable


def draw_N_C_Acc(accuracy, N, C, k, isfull, path):
    """
    draws three dimensional plot for N,C versus accuracy
    :param accuracy: dictionary of tuple to float. Tuples are formed as (k, N, C, isfull)
    :param N: List of different n values to be drawn on the plot
    :param C: List of different c values to be drawn on the plot
    :param k: a single k value to ve drawn on the plot
    :param isfull: whether plot the full sketches or partials
    :param path: path to save the resuls
    """
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
    plt.title('Accuracy Contour Plot for different N and C for for %s' % 'Full sketches' if isfull else 'Partial sketches')
    fig.savefig(path + '/' + 'draw_N_C_Acc_%s.png' % ('Full' if isfull else 'Partial'))

def draw_N_C_Reject(delay_rate, N, C, k, isfull, path):
    """
    draws three dimensional plot for N,C versus reject rate
    :param delay_rate: dictionary of tuple to float. Tuples are formed as (k, N, C, isfull)
    :param N: List of different n values to be drawn on the plot, must exists on the dictionary
    :param C: List of different c values to be drawn on the plot, must exists on the dictionary
    :param k: a single k value to ve drawn on the plot
    :param isfull: whether plot the full sketches or partials
    :param path: path to save the resuls
    """
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

def draw_N_C_Reject_Contour(reject_rate, N, C, k, isfull, path):
    """
    draws two dimensional contour plot for N,C versus reject rate
    :param reject_rate: dictionary of tuple to float. Tuples are formed as (k, N, C, isfull)
    :param N: List of different n values to be drawn on the plot, must exists on the dictionary
    :param C: List of different c values to be drawn on the plot, must exists on the dictionary
    :param k: a single k value to ve drawn on the plot
    :param isfull: whether plot the full sketches or partials
    :param path: path to save the resuls
    """

    fig = plt.figure()
    nmesh, cmesh = np.meshgrid(N, C)

    nlist, clist, rejlist =[], [], []
    for n in N:
        for c in C:
            nlist.append(n)
            clist.append(c)
            rejlist.append(reject_rate[k, n, c, isfull])

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

    plt.title('Reject Rate Contour Plot for different N and C for for %s' % 'Full sketches' if isfull else 'Partial sketches')
    plt.xlabel('N')
    plt.ylabel('C')
    plt.colorbar()
    plt.clim(0, 100)

    fig.savefig(path + '/' + 'draw_N_C_Reject_Contour_%s.png' % ('Full' if isfull else 'Partial'))

def draw_N_C_Acc_Contour(accuracy, N, C, k, isfull, path):
    """
    draws two dimensional contour plot for N,C versus accuracy
    :param accuracy: dictionary of tuple to float. Tuples are formed as (k, N, C, isfull)
    :param N: List of different n values to be drawn on the plot, must exists on the dictionary
    :param C: List of different c values to be drawn on the plot, must exists on the dictionary
    :param k: a single k value to ve drawn on the plot
    :param isfull: whether plot the full sketches or partials
    :param path: path to save the resuls
    """
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

    plt.title('Accuracy Contour Plot for different N and C for for %s' %'Full sketches' if isfull else 'Partial sketches')
    plt.xlabel('N')
    plt.ylabel('C')
    plt.colorbar()

    fig.savefig(path + '/' + 'draw_N_C_Acc_Contour_%s.png' % ('Full' if isfull else 'Partial'))


def draw_N_C_Rej_Contour_Low(reject_rate, N, C, k, isfull, path, title=''):
    """
    draws two dimensional contour plot for N,C versus accuracy
    :param accuracy: dictionary of tuple to float. Tuples are formed as (k, N, C, isfull)
    :param N: List of different n values to be drawn on the plot, must exists on the dictionary
    :param C: List of different c values to be drawn on the plot, must exists on the dictionary
    :param k: a single k value to ve drawn on the plot
    :param isfull: whether plot the full sketches or partials
    :param path: path to save the resuls
    """
    fig = plt.figure(figsize=(12, 12))
    C = np.linspace(0,100,11)
    nmesh, cmesh = np.meshgrid(N, C)
    accmesh = []
    for n in N:
        temp = []
        for c in C:
            temp.extend([reject_rate[(k,n,c,isfull)]])
        accmesh.append(temp)

    numxTicks = min(10, max(N))
    plt.xticks(np.arange(0, 11, 1), np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, max(N)+1, 1),np.arange(1, max(N)+1, 1))

    plt.imshow(accmesh, interpolation='none', cmap='summer')

    plt.title('Reject Rate for  %s %s' %(('Full sketches' if isfull else 'Partial sketches'),title))
    plt.xlabel('C')
    plt.ylabel('N')
    #plt.grid(ls='solid')

    for n in N:
        for c in C:
            plt.text(c/10-0.2, n-1, '%.2f' % reject_rate[(k,n,c,isfull)], fontsize=10)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(cax=cax)
    plt.clim(0, 100)
    #plt.show()
    ''''
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
    numxTicks = min(10, max(N))

    plt.xticks(np.linspace(0, len(nlist), numxTicks), [int(f) for f in np.linspace(1, max(N), numxTicks)])
    plt.yticks(np.linspace(0, len(acclist), 11), [int(f) for f in np.linspace(1, max(C), 10)])

    plt.title('Accuracy Contour Plot for different N and C for for %s' %'Full sketches' if isfull else 'Partial sketches')
    plt.xlabel('C')
    plt.ylabel('N')
    plt.colorbar()
    '''
    fig.savefig(path + '/' + 'draw_N_C_Rej_Contour_%s.png' % ('Full' if isfull else 'Partial'))

def draw_N_C_Acc_Contour_Low(accuracy, N, C, k, isfull, path, title=''):
    """
    draws two dimensional contour plot for N,C versus accuracy
    :param accuracy: dictionary of tuple to float. Tuples are formed as (k, N, C, isfull)
    :param N: List of different n values to be drawn on the plot, must exists on the dictionary
    :param C: List of different c values to be drawn on the plot, must exists on the dictionary
    :param k: a single k value to ve drawn on the plot
    :param isfull: whether plot the full sketches or partials
    :param path: path to save the resuls
    """
    fig = plt.figure(figsize=(12, 12))
    C = np.linspace(0,90,10)
    nmesh, cmesh = np.meshgrid(N, C)
    accmesh = []
    for n in N:
        temp = []
        for c in C:
            temp.extend([accuracy[(k,n,c,isfull)]])
        accmesh.append(temp)

    numxTicks = min(10, max(N))
    plt.xticks(np.arange(0, 11, 1), np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, max(N)+1, 1),np.arange(1, max(N)+1, 1))

    plt.imshow(accmesh, interpolation='none', cmap='summer')

    plt.title('Accuracy for  %s %s' %(('Full sketches' if isfull else 'Partial sketches'),title))
    plt.xlabel('C')
    plt.ylabel('N')
    #plt.grid(ls='solid')

    for n in N:
        for c in C:
            plt.text(c/10-0.2, n-1, '%.2f' % accuracy[(k,n,c,isfull)], fontsize=10)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.2)
    plt.colorbar(cax=cax)
    plt.clim(0, 100)
    #plt.show()
    ''''
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
    numxTicks = min(10, max(N))

    plt.xticks(np.linspace(0, len(nlist), numxTicks), [int(f) for f in np.linspace(1, max(N), numxTicks)])
    plt.yticks(np.linspace(0, len(acclist), 11), [int(f) for f in np.linspace(1, max(C), 10)])

    plt.title('Accuracy Contour Plot for different N and C for for %s' %'Full sketches' if isfull else 'Partial sketches')
    plt.xlabel('C')
    plt.ylabel('N')
    plt.colorbar()
    '''
    fig.savefig(path + '/' + 'draw_N_C_Acc_Contour_%s.png' % ('Full' if isfull else 'Partial'))

def draw_Completeness_Accuracy(accuracy_fullness, fullness, k, n, C, isfull, path):
    fig = plt.figure(figsize=(12, 12))

    x = fullness
    # to make legend more appropriate
    for c in C[::-1]:
        y = []
        for f in fullness:
            succ = accuracy_fullness[(k, n, c, isfull, f, True)]
            fail = accuracy_fullness[(k, n, c, isfull, f, False)]

            perc = (float(succ)/(succ+fail))*100 if (succ+fail)!=0 else 0
            y.extend([perc])
        plt.plot(x,y, label='C=%.1f'%(float(c)/100))

    plt.xlim([0, 13])
    plt.ylim([0,105])

    plt.xticks(np.linspace(0,12,13))
    plt.yticks(np.linspace(0,100,11))

    plt.ylabel('Avg. recognutuin rate(%)')
    plt.xlabel('Completeness range[(k-1)*10%, k*10%) - last col. implies full sketches')
    plt.title('Avg. prediction acuracy rate over completeness (N='+str(n)+')')
    plt.grid()
    plt.legend(loc='lower right')

    fig.savefig(path + '/' + 'draw_Completeness_Accuracy_%s.png' % ('Full' if isfull else 'Partial'))

def draw_Reject_Acc(Accuracy, Reject_rate, N, k, isfull, labels, path):
    """
    Draws two dimensional ROC-like curve, reject rate versus accuracy for multiple results
    It enables to compare several implementations on an objective basis
    :param Accuracy: list of accuracy dictionaries
    :param Reject_rate: list of reject rate dictionaries
    :param N: List of different n values to be drawn on the plot, must exists on the dictionary
    :param k: a single k value to ve drawn on the plot
    :param isfull: whether plot the full sketches or partials
    :param labels: legend strings for different implementations, with the same order as dictionaries
    :param path: path to save the resuls
    :return:
    """
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
                x_reject.append(Reject_rate[i][(k, n, c, isfull)])
                y_acc.append(Accuracy[i][(k, n, c, isfull)])

            pylab.scatter(x_reject, y_acc, alpha=1, s=100, color=color[i], marker=markers[n+i], label=labels[i] + ' N='+str(n))
            miny= min (min(y_acc),miny)
        # labels
    pylab.legend(loc='lower right')
    pylab.xlabel('Reject Rate')
    pylab.ylabel('Accuracy')
    pylab.xlim([0,90])
    pylab.ylim([80,100+5])
    pylab.title('Performance Comparison on %s Symbols Using %i Clusters' % ('Full' if isfull else 'Partial',k))

    plt.savefig(path + '/' + 'draw_Reject_Acc_%s.png' % ('Full' if isfull else 'Partial'))

def draw_n_Acc(accuracy, c, k, isfull, reject_rate, path):
    """
    draws a two dimensional slice of N,C versus accuracy plot in a nice format
    :param accuracy: dictionary of tuple to float. Tuples are formed as (k, N, C, isfull)
    :param c: a single threshold to be drawn. Determines the slice of N,C versus accuracy plot
    :param k: a single number-of-clusters parameter to be drawn
    :param isfull: whether plot the full sketches or partials
    :param reject_rate: dictionary of tuple to float. Tuples are formed as (k, N, C, isfull)
    :param path: path to save the resuls
    :return:
    """
    fig = plt.figure()
    maxN = max(key[1] for key in accuracy.keys())

    x, y = [], []
    for n in range(1, maxN + 1):
        if (k, n, c, isfull) in accuracy.keys():
            x.append(n)
            y.append(accuracy[(k, n, c, isfull)])
            plt.annotate('N:%i,Acc:%i%%,Rej:%i%%' % (n, accuracy[(k, n, c, isfull)], reject_rate[(k, n, c, isfull)]), \
                         (n + 0.1, accuracy[(k, n, c, isfull)] + 0.1))

    # horizontal lines
    plt.plot([1 - 1, maxN + 1], [60, 60], 'r--')
    plt.plot([1 - 1, maxN + 1], [80, 80], 'g--')

    # labels
    plt.xlabel('N')
    plt.ylabel('Accuracy')
    plt.title('C:%i k:%i for %s' % (c, k, 'Full sketches' if isfull else 'Partial sketches'))

    plt.ylim(min(max(0, min(y)-10),40), 100 * 1.1)
    plt.xlim(1 - 1, 10)
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