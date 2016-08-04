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
from Predictor import *
import os
import time
import classesFile

from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
predictor = None
global myfiles
myfiles = classesFile.files

def trainIt(trainingName, trainingpath, numclass, numfull, numpartial, k, files):
        fio = FileIO()
        extr = Extractor('../data/')
        features, isFull, classId, names, folderList = extr.loadfolders(  numclass   = numclass,
                                                                          numfull    = numfull,
                                                                          numpartial = numpartial,
                                                                          folderList = files)
        constarr = getConstraints(size=len(features), isFull=isFull, classId=classId)
        ckmeans = CKMeans(constarr, np.transpose(features), k)
        kmeansoutput = ckmeans.getCKMeans()

        # find heterogenous clusters and train svm
        trainer = Trainer(kmeansoutput, classId, features)
        heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
        fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                         trainingpath, trainingName)
        trainer.trainSVM(heteClstrFeatureId, trainingpath)
        return kmeansoutput, classId

@app.route("/", methods=['POST','GET'])
def handle_data():
    global predictor
    timeStart = time.time()
    try:
        queryjson = request.args.get('json')
        print 'Server responded in %.3f seconds' % float(time.time()-timeStart)
        return predictor.giveOutput(queryjson,5)
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
    global myfiles
    ForceTrain = False
    numclass, numfull, numpartial = 10, 6, 3
    k = numclass
    trainingName = '%s__CFPK_%i_%i_%i_%i' % ('zz', numclass, numfull, numpartial, k)
    trainingpath = '../data/training/' + trainingName
    fio = FileIO()

    # if training data is already computed, import
    if os.path.exists(trainingpath) and not ForceTrain:
        names, classId, isFull, features, kmeansoutput, myfiles = fio.loadTraining(trainingpath + "/" + trainingName)
    else:
        kmeansoutput,classId = trainIt(trainingName, trainingpath, numclass, numfull, numpartial, k, myfiles)
    global predictor
    predictor = Predictor(kmeansoutput, classId, trainingpath, myfiles)

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run(host='0.0.0.0', debug=False)
    print 'Server ended'
if __name__ == '__main__':main()
