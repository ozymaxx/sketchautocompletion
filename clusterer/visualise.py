import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


cl1 = np.array([0, 1, 2,4,6])
cl2 = np.array([3,5,7,9])
cl3 = np.array([8,10])
clusters = [cl1,cl2,cl3]
features = np.array([[1, 2], [2, 4], [5, 3], [1, 1], [3, 9], [2, 5], [8, 3], [3, 2],[2,2],[3,5],[4,4]])
labels = np.array([1,3,2,3,2,1,2,2,1,2,3])

t1 = (cl1,[1,2])
t2 = (cl2,[3,3])
t3 = (cl3, [2,3])

out1 = [(cl1,cl2,cl3),([2,2],[2,2],[2,2])]
out = [t1,t2,t3]


def visualiseBeforeClustering(out):
    color = 'black'
    for cluster in out[0]:
        for i in cluster:
              x = features.tolist()[i]
              scale = 80
              plt.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black')
    plt.figure()
    plt.grid(True)

def visualiseAfterClustering(out):
    colorList = cm.rainbow(np.linspace(0, 1, len(out[0])))
    index = 0
    for cluster in out[0]:
        index+=1
        color = colorList[index-1]
        for i in cluster:
              x = features.tolist()[i]
              scale = 80


              plt.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black')

    plt.grid(True)
    # plt.show()

visualiseBeforeClustering(out1)
visualiseAfterClustering(out1)
plt.show()