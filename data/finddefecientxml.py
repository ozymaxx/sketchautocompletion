from os import listdir
from os.path import isfile, join
import sys
sys.path.append('../predict/')
sys.path.append('../data/')
from FileIO import *
import os

def isSketchFull(name):
    return not name[-1].isdigit()

symbolspat = 'C:/Users/1003/Desktop/nicicon/symbols'
imagefolders = [[symbolspat + '/' + f + '/eval/', symbolspat + '/' + f + '/test/', symbolspat + '/' + f + '/train/']
                for f in listdir(symbolspat) if not isfile(join(symbolspat, f))]

# concatenate all inner lists
imagefolders = [x for y in imagefolders for x in y]
imagefolders.sort()
fio = FileIO()
missingNames = []
for folder in imagefolders:
    # get the images
    imgpaths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    imgpaths.sort()
    imagenames = [imgpath[imgpath.rfind('/')+1:-4] for imgpath in imgpaths]

    folderspl = folder.split('/')
    className = folderspl[-3]
    type = folderspl[-2] # train, eval or test

    csvname =className + '_' + type + '.csv'#car-train.csv
    names, isFull, features = fio.load('C:/Users/1003/Desktop/nicicon/csv/' + type + '/' + className + '/' + csvname)
    missingNames.extend([folder  + imgname + '.xml' for imgname in imagenames if imgname not in names])

    import os
print missingNames


    #pathtowrite = 'C:/Users/1003/Desktop/nicicon/csv/' + className + '/'
    #fio.save(isFull, names, features, 'C:/Users/1003/Desktop/nicicon/csv/' + type + '/' + className + '/' + csvname)



