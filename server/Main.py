import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../test/')
sys.path.append("../../libsvm-3.21/python/")
from extractor import *
import numpy as np
from featureutil import *
from FeatureExtractor import *
from FileIO import *
from shapecreator import *
import time

class MainHelper:
    """The main class to be called"""
    def __init__(self):
        self.features = list()
        self.classId = list()
        self.isFull = list()
        self.path = '../json/'
        self.trainer = None
        self.kmeansoutput = None
        self.priorClusterProb = None
        self.d = {}
        self.names = None
        self.files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack', 'banana', 'barn',
                      'baseball-bat', 'basket']
        self.extr = Extractor('../data/')
        self.extr.prnt = True
        self.n = 10

    def getBestPredictions(self, classProb):
        a = sorted(classProb, key=classProb.get, reverse=True)[:self.n]
        l = ''
        l1 = ''
        for i in a:
            l1 += str(classProb[i])
            l += self.files[i]
            l += '&'
            l1+='&'
        l1 = l1[:-1]
        return l+l1

    def trainIt(self, numClass, numFull, numPartial, k):
        self.features, self.isFull, self.classId, self.names, self.folderList = self.extr.loadfolders(numclass = numClass, numfull=numFull, numpartial=numPartial,
                                                                                                      folderList=self.files)
        print 'Loaded ' + str(len(self.features)) + ' sketches'

        # train constrained k-means
        NUMPOINTS = len(self.features)
        constarr = getConstraints(NUMPOINTS, self.isFull, self.classId)
        ckmeans = CKMeans(constarr, np.transpose(self.features), k=k)
        self.kmeansoutput = ckmeans.getCKMeans()

        # find heterogeneous clusters and train svm
        self.trainer = Trainer(self.kmeansoutput, self.classId, self.features)
        heteClstrFeatureId, heteClstrId = self.trainer.getHeterogenous()
        self.trainer.trainSVM(heteClstrFeatureId)

    def predictByInstance(self, instance):
        # find the probability of given feature to belong any of trained classes
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb)
        return self.getBestPredictions(classProb)

    def predictByPath(self,fullsketchpath):
        instance = featureExtract(fullsketchpath)
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb)
        return self.getBestPredictions(classProb)

    def predictByString(self, jstring):
        loadedSketch = shapecreator.buildSketch('json', jstring)
        featextractor = IDMFeatureExtractor()
        instance = featextractor.extract(loadedSketch)
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb)
        return self.getBestPredictions(classProb)

    def saveTraining(self, f):
        fio = FileIO()
        fio.saveTraining(self.names,self.classId, self.isFull, self.features, self.kmeansoutput, f)

    def loadTraining(self, f):
        fio = FileIO()
        self.names, self.classId, self.isFull, self.features, self.kmeansoutput = fio.loadTraining(f)

        # find heterogenous clusters and train svm
        self.trainer = Trainer(self.kmeansoutput, self.classId, self.features)  ### FEATURES : TRANSPOSE?
        heteClstrFeatureId, heteClstrId = self.trainer.getHeterogenous()
        self.trainer.trainSVM(heteClstrFeatureId)


from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
m = MainHelper()

@app.route("/", methods=['POST','GET'])
def handle_data():
    global m
    timeStart = time.time()
    if request.method == 'POST':
        try:
            print(type(json.dumps(request.json)))
            queryjson = request.json
            print(json.dumps(request.json))
            text_file = open('./query.json', "w")
            text_file.write(json.dumps(request.json))
            text_file.close()

            answer = m.predictByPath("./query.json")
            return answer
        except Exception as e:
            flash(e)
            text_file = open('./errorquery.json', "w")
            text_file.write(json.dumps(request.json))
            text_file.close()
            print(e)
            return "Error" + str(e)
    else:
        try:
            queryjson = request.args.get("json")
            answer = m.predictByString(str(queryjson))

            print 'Server responded in %.3f seconds' % float(time.time()-timeStart)
            return answer

        except Exception as e:
            flash(e)
            print(request.values)
            return "Error" + str(e)


@app.route("/send", methods=['POST','GET'])
def return_probables():
    print 'RETURN PROBABILITIES'
    try:
        if (request.method == 'POST'):
            #os.system("python run.py");
            return "airplane&angel&arm&banana&bell"

        else:
            return "airplane&angel&arm&banana&bell"
            #"Hello World - you sent me a GET " + str(request.values)
    except Exception as e:
        flash(e)
        return "Error" + str(e)

@app.route("/home", methods=['GET'])
def homepage():
    print 'SEE HOMEPAGES'
    return render_template("index.html")

def main():
    doTrain = True
    numclass = 10
    numfull = 5
    numpartial = 10
    k = numclass

    if doTrain:
        m.trainIt(numclass, numfull, numpartial, k)
        m.saveTraining("lol")
    else:
        m.loadTraining("lol")

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run(host='0.0.0.0', debug= False)

if __name__ == '__main__':main()