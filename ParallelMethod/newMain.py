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
import time
import numpy as np
import classesFile
import math

from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
predictor = None
files = classesFile.files
normalProb = []

my_numclass = 15
my_numfull = 10
my_numpartial= 5
my_k = my_numclass

def getBestPredictions(classProb, n):
        a = sorted(classProb, key=classProb.get, reverse=True)[:n]
        l = ''
        l1 = ''
        for i in a:
            l1 += str(classProb[i])
            l += i
            l += '&'
            l1 += '&'
        l1 = l1[:-1]
        return l + l1

def newTraining(n,files, numclass, numfull,numpartial,k, name):
    extr = Extractor('../data/')
    fio = FileIO()
    global normalProb

    import os
    path = '../data/newMethodTraining/' + name
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(n/5):
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
        normalProb.append(totalNumOfInstances)

def PredictIt(n, q, name, numclass, numfull, numpartial,k):
    global normalProb
    kj = 0
    for i in normalProb:
        kj += i
    for i in range(len(normalProb)):
        normalProb[i] = float(normalProb[i])/kj

    extr = Extractor('../data/')
    fio = FileIO()

    l = []
    a = newMethodPredictor()

    for i in range(n/5):
        trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
        trainingpath = '../data/newMethodTraining/' + name+ '/' + trainingName
        b = fio.loadOneFeature(trainingpath +'/' + str(i) + "__Training_Center_")
        l.append(b)

    savingProbs = a.probToSavings(q, l, normalProb)
    savingProbs = [x/sum(savingProbs) for x in savingProbs]

    out = dict()
    for i in range(n/5):
        trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
        trainingpath = '../data/newMethodTraining/' + name+ '/' + trainingName
        names, classId, isFull, features, kmeansoutput, loadedFolders = fio.loadTraining(trainingpath + "/" + trainingName)

        predictor = newMethodPredictor(kmeansoutput, classId, trainingpath,loadedFolders)

        a = predictor.predictByString(q)
        for m in a.keys():
            out[m] = a[m]*savingProbs[i]
    return out


@app.route("/", methods=['POST','GET'])
def handle_data():
    global predictor,my_numclass,my_k,my_numfull,my_numpartial
    timeStart = time.time()
    try:
        queryjson = request.args.get('json')
        classProb = PredictIt(my_numclass,queryjson, 'hello',my_numclass,my_numfull,my_numpartial,my_k)
        print 'Server responded in %.3f seconds' % float(time.time()-timeStart)
        return getBestPredictions(classProb, 5)
    except Exception as e:
        flash(e)
        print(request.values)
        return "Error" + str(e)

@app.route("/send", methods=['POST','GET'])
def return_probabilities():
    print 'SERVER: return_probabilities method called'
    raise NotImplementedError

@app.route("/home", methods=['GET'])
def homepage():
    print 'SERVER: homepage method called'
    return render_template("index.html")

def main():
    ForceTrain = True
    global my_numclass, my_numfull, my_numpartial, my_k
    n = my_numclass

    # if training data is already computed, import
    if not ForceTrain:
        pass
        # names, classId, isFull, features, kmeansoutput = fio.loadTraining(trainingpath + "/" + trainingName)
    else:
        newTraining(n , files, my_numclass, my_numfull, my_numpartial, my_k, 'hello')

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run(host='0.0.0.0', debug=False)
    print 'Server ended'
if __name__ == '__main__':main()
