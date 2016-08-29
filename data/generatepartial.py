from os import listdir
from os.path import isfile, join
"""
Generate partial sketches from full .xml sketches
"""
symbolspat = '../nicicon/symbols/'
imagefolders = [[join(symbolspat, f) + '/eval/', join(symbolspat, f) + '/test/', join(symbolspat, f) + '/train/']
                for f in listdir(symbolspat) if not isfile(join(symbolspat, f))]

# concatenate all inner lists
imagefolders = [x for y in imagefolders for x in y]

for folder in imagefolders:
    # get the images
    imgpaths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    for imgpath in imgpaths:
        file = open(imgpath, 'r')
        imgxml = file.read()

        # find all occurrences of 'stroke'
        import re
        allStrokeIndex = [m.start() for m in re.finditer('stroke', imgxml)]
        strokeIndices = [[0,0] for x in range(len(allStrokeIndex)/2)]

        for idx in range(len(strokeIndices)):
            strokeIndices[idx][0] = allStrokeIndex[2*idx]-1 #<stroke
            strokeIndices[idx][1] = allStrokeIndex[2*idx+1] + 6 #</stroke>
        # found starting and ending indices of all strokes

        # for every stroke create a new partial
        for partialIdx in range(1, len(strokeIndices)):
            partialimg = imgxml[0:strokeIndices[partialIdx][0]] + '</sketch>'
            partialpath = imgpath[0:len(imgpath)-4] + '_' + str(partialIdx) + '.xml'

            text_file = open(partialpath, "w")
            text_file.write(partialimg)
            text_file.close()
