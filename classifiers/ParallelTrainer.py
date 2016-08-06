import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../test/')
sys.path.append('../data/')
sys.path.append("../../libsvm-3.21/python")
from extractor import *
from FileIO import *
from newMethodPredictor import *
import numpy as np
import classesFile

class newTrainer:
    """Trainer Class used for the training"""

    def __init__(self, n):
        self.n = n
        self.trainingAdress = '../data/newMethodTraining/'
        self.files = classesFile.files

    def trainSWM(self, numclass, numfull, numpartial, k, name):

        files = self.files
        n = self.n
        extr = Extractor('../data/')
        fio = FileIO()
        global normalProb

        import os
        path = self.trainingAdress + name
        if not os.path.exists(path):
            os.mkdir(path)

        for i in range(numclass/n):
            trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
            trainingpath = path +'/'+ trainingName
            features, isFull, classId, names, folderList = extr.loadfolders(  numclass   = numclass,
                                                                              numfull    = numfull,
                                                                              numpartial = numpartial,
                                                                              folderList = files[i*5:(i+1)*5])
            constarr = getConstraints(size=len(features), isFull=isFull, classId=classId)
            ckmeans = CKMeans(constarr, np.transpose(features), k)
            kmeansoutput = ckmeans.getCKMeans()

            # find heterogenous clusters and train svm
            trainer = Trainer(kmeansoutput, classId, features)
            heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
            trainer.trainSVM(heteClstrFeatureId, trainingpath)
            fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                             trainingpath, trainingName)
            trainer.trainSVM(heteClstrFeatureId, trainingpath)

            nowCenter = np.zeros(len(features[0]))
            totalNumOfInstances = len(features)
            for cluster in kmeansoutput[0]:
                for instance in cluster:
                    nowCenter += features[instance]
            nowCenter = nowCenter/totalNumOfInstances
            fio.saveOneFeature(trainingpath +'/' + str(i) + "__Training_Center_",nowCenter)
            # normalProb.append(totalNumOfInstances)


