import numpy as np
def partitionfeatures(features, isFull, classId, names, numtestdata, selectTestRandom = True):
    numclass = len(set(classId))
    testfeatures, testnames, testclassid, testisFull  = [], [], [], []





    for j in range(numclass):
        for i in range(numtestdata):
            # take first partial sketch to be removed

            # select test data randomly
            # at most numtestdata for each class

            index = 0
            if selectTestRandom:
                index = np.random.randint(0, len(features))
                while testclassid.count(classId[index]) >= numtestdata:
                    index = np.random.randint(0, len(features))
            else:
                while testclassid.count(classId[index]) >= numtestdata:
                    index += 1

            fullindex = names[index].split('_')[0] + '_' + names[index].split('_')[1]
            classname = names[index].split('_')[0]
            sketchid = names[index].split('_')[1]

            # get test data mask
            cond = [names[partialindex].split('_')[0] == classname and names[partialindex].split('_')[1] == sketchid for partialindex in range(len(features))]

            # get only the full sketches for test data
            testfeatures.extend([features[index] for index in range(len(features)) if cond[index]])
            testisFull.extend([isFull[index] for index in range(len(isFull)) if cond[index]])
            testnames.extend([names[index] for index in range(len(names)) if cond[index]])
            testclassid.extend([classId[index] for index in range(len(classId)) if cond[index]])

            # remove any partial sketch from the traning data
            features = [features[index] for index in range(len(features)) if not cond[index]]
            names = [names[index] for index in range(len(names)) if not cond[index]]
            isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
            classId = [classId[index] for index in range(len(classId)) if not cond[index]]

    return train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classid

def partitionfeatures_notin(features, isFull, classId, names, numtestdata, trainnames, selectTestRandom = True):
    numclass = len(set(classId))

    testfeatures = [features[index] for index in range(len(features)) if names[index] not in trainnames]
    testisFull = [isFull[index] for index in range(len(isFull)) if names[index] not in trainnames]
    testclassid = [classId[index] for index in range(len(classId)) if names[index] not in trainnames]
    testnames = [names[index] for index in range(len(names)) if names[index] not in trainnames]

    features = [features[index] for index in range(len(features)) if names[index] in trainnames]
    isFull = [isFull[index] for index in range(len(isFull)) if names[index] in trainnames]
    classId = [classId[index] for index in range(len(classId)) if names[index] in trainnames]
    names = [names[index] for index in range(len(names)) if names[index] in trainnames]

    if len(testfeatures) < numtestdata:
        raise Warning('not enough test data')

    if selectTestRandom:
        import random
        testfeatures = random.sample(testfeatures, numtestdata)
        testisFull = random.sample(testisFull, numtestdata)
        testclassid = random.sample(testclassid, numtestdata)
        testnames = random.sample(testnames, numtestdata)
    else:
        testfeatures = testfeatures[:numtestdata]
        testisFull = testisFull[:numtestdata]
        testclassid = testclassid[:numtestdata]
        testnames = testnames[:numtestdata]

    return features, isFull, classId, names, testfeatures, testisFull, testnames, testclassid


    def processName(name):
        classname = names[index].split('_')[0]
        sketchid = names[index].split('_')[1]
        return