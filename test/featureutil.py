import numpy as np
def partitionfeatures(features, isFull, classId, names, numtestdata, randomPartioning = True):
    numclass = len(set(classId))
    testfeatures = list()
    testnames = list()
    testclassid = list()

    for j in range(numclass):
        for i in range(numtestdata):
            # take first partial sketch to be removed

            # select test data randomly
            # at most numtestdata for each class

            index = 0
            if randomPartioning:
                index = np.random.randint(0, len(features))
                while testclassid.count(classId[index]) >= numtestdata:
                    index = np.random.randint(0, len(features))
            else:
                while testclassid.count(classId[index]) >= numtestdata:
                    index += 1

            fullindex = names[index].split('_')[0] + '_' + names[index].split('_')[1]
            namesplit1 = names[index].split('_')[0]
            namesplit2 = names[index].split('_')[1]

            # get test data mask
            cond = [names[partialindex].split('_')[0] == namesplit1 and names[partialindex].split('_')[1] == namesplit2 for partialindex in range(len(features))]

            # get only the full sketches for test data
            testfeatures.extend([features[index] for index in range(len(features)) if cond[index] and isFull[index]])
            testnames.extend([names[index] for index in range(len(features)) if cond[index] and isFull[index]])
            testclassid.extend([classId[index] for index in range(len(features)) if cond[index] and isFull[index]])

            # remove any partial sketch from the traning data
            features = [features[index] for index in range(len(features)) if not cond[index]]
            names = [names[index] for index in range(len(names)) if not cond[index]]
            isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
            classId = [classId[index] for index in range(len(classId)) if not cond[index]]

    return features, isFull, classId, names, testfeatures, testnames, testclassid