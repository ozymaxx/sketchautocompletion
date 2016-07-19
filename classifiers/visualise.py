import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.markers as mark
import numpy as np



def visualiseAfterClustering(modelPoints,out, features, classId,  isFull, centers, title):
    def getMarkerList():
        numClass = len(set(classId))
        marker_list = list(mark.MarkerStyle.filled_markers)
        markIndex = 4
        while numClass >= len(marker_list):
            marker_list.append((markIndex,1))
            markIndex+=1

        return marker_list

    fig1 = plt.figure()
    fig1.canvas.set_window_title(str(title))
    ax1 = fig1.add_subplot(111)

    fig2 = plt.figure()
    fig2.canvas.set_window_title("K="+str(title) + " Full Sketch")
    ax2 = fig2.add_subplot(111)

    colorList = cm.rainbow(np.linspace(0, 1, len(out[0])))
    #print len(centers[0])
    index = 0
    marker_list = getMarkerList()

    count = 0

    for i in range(len(centers[0])):
             ax1.scatter(centers[0][i], centers[1][i], c='#000000', s=300,
                        alpha=0.5, edgecolors='black')

    for m in modelPoints:
        for i in m:
            ax1.scatter(i[1], i[2], c='black', s=500,
                        alpha=0.5, edgecolors='black', marker = "+")

    for cluster in out[0]:
        index+=1
        color = colorList[index-1]
        ax1.scatter(out[1][index-1][0], out[1][index-1][1], c='red', s=300, label=color,
                    alpha=0.5, edgecolors='black' )

        for i in cluster.astype(int):
              x = features.tolist()[i]
              scale = 80
              marker = classId[i]

              ax1.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black', marker= marker_list[marker])

              if(isFull[i] == 1):
                count += 1
                ax2.scatter(x[0], x[1], c=color, s=scale, label=color,
                    alpha=0.5, edgecolors='black', marker= marker_list[marker])

    print count
    plt.grid(True)
    ax1.grid(True)
