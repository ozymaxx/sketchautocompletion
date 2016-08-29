from os import listdir
from os.path import isfile, join
import sys
"""
Apply feature extraction to raw sketch files, and save them to a single csv file
"""
sys.path.append('../predict/')
from feature import *
from FileIO import *
import os

def isSketchFull(name):
    return not name[-1].isdigit()

symbolspat = 'C:/Users/1003/Desktop/nicicon/symbols'
imagefolders = [[symbolspat + '/'+ f + '/eval/', symbolspat + '/'+ f + '/test/', symbolspat + '/'+ f + '/train/']
                for f in listdir(symbolspat) if not isfile(join(symbolspat, f))]

# concatenate all inner lists
imagefolders = [x for y in imagefolders for x in y]
imagefolders.sort()
fio = FileIO()

for folder in imagefolders:
    # get the images
    imgpaths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    imgpaths.sort()

    folderspl = folder.split('/')
    className = folderspl[-3]
    type = folderspl[-2] # train, eval or test

    csvname =className + '_' + type + '.csv' #car-train.csv

    features, isFull, names = [], [], []
    for imgpath in imgpaths:
        #print 'Writing ' + imgpath
        file = open(imgpath, 'r')
        imgxml = file.read()

        idx = imgpath.rfind("/")
        name = imgpath[idx+1:]
        name = name[:len(name)-4]
        try:
            loadedSketch = shapecreator.buildSketch('xml', imgxml)
            featextractor = IDMFeatureExtractor()
            imgxmlFeature = featextractor.extract(loadedSketch)
        except:
            print 'ERROR ON ' + imgpath
            continue

        features.append(imgxmlFeature)
        names.append(name)
        isFull.append(isSketchFull(name))

    #pathtowrite = 'C:/Users/1003/Desktop/nicicon/csv/' + className + '/'
    #fio.save(isFull, names, features, 'C:/Users/1003/Desktop/nicicon/csv/' + type + '/' + className + '/' + csvname)



