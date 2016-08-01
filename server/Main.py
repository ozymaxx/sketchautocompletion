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

class M:
    """The main class to be called"""
    def __init__(self, trainingName):
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
        self.numclass = None
        self.numfull = None
        self.numpartial = None
        self.k = None
        self.mainDirectory = "../savedTrainings/"
        self.trainingName = trainingName
        self.subDirectory = self.mainDirectory + trainingName

    def getBestPredictions(self, c):
        a = sorted(c, key=c.get, reverse=True)[:self.n]
        l = ''
        l1 = ''
        for i in a:
            l1 += str(c[i])
            l += self.files[i]
            l += '&'
            l1+='&'
        l1 = l1[:-1]
        return l+l1

    def trainIt(self, numClass, numFull, numPartial, k):

        self.k=k
        self.numclass = numClass
        self.numfull = numFull
        self.numpartial = numPartial

        import os
        self.subDirectory = self.mainDirectory + self.trainingName + "__" + "CFPK:" + str(self.numclass) + "_" + str(self.numfull) + "_" + str(self.numpartial) + "_" + str(self.k)

        if not os.path.exists(self.subDirectory):
            os.makedirs(self.subDirectory)

        self.features, self.isFull, self.classId, self.names, self.folderList = self.extr.loadfolders(numclass = numClass, numfull=numFull, numpartial=numPartial,
                                                                                                      folderList=self.files)
        print 'Loaded ' + str(len(self.features)) + ' sketches'

        #partition data into test and training
        # train constrained k-means

        NUMPOINTS = len(self.features)
        constarr = getConstraints(NUMPOINTS, self.isFull, self.classId)
        ckmeans = CKMeans(constarr, np.transpose(self.features), k=k)
        self.kmeansoutput = ckmeans.getCKMeans()

        # find heterogenous clusters and train svm
        self.trainer = Trainer(self.kmeansoutput, self.classId, self.features)  ### FEATURES : TRANSPOSE?
        heteClstrFeatureId, heteClstrId = self.trainer.getHeterogenous()
        self.trainer.trainSVM(heteClstrFeatureId, self.subDirectory)

    def predictIt(self, instance):
        # find the probability of given feature to belong any of athe classes
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb,self.subDirectory)
        return self.getBestPredictions(classProb)

    def predictByPath(self,fullsketchpath):

        instance = featureExtract(fullsketchpath)
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb,self.subDirectory)
        return self.getBestPredictions(classProb)
    def predictByString(self, jstring):
        loadedSketch = shapecreator.buildSketch('json', jstring)
        featextractor = IDMFeatureExtractor()
        instance = featextractor.extract(loadedSketch)
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb,self.subDirectory)
        return self.getBestPredictions(classProb)

    def saveTraining(self):
        import os
        if not os.path.exists(self.subDirectory):
            os.makedirs(self.subDirectory)
        fio = FileIO()
        fio.saveTraining(self.names,self.classId, self.isFull, self.features, self.kmeansoutput,
                         self.subDirectory+"/" + self.trainingName)

    def loadTraining(self,f):

        self.subDirectory = self.mainDirectory + f
        name = f.split("__")[0]
        fio = FileIO()
        self.names, self.classId, self.isFull, self.features, self.kmeansoutput = fio.loadTraining(self.subDirectory + "/" + name)

        # find heterogenous clusters and train svm
        self.trainer = Trainer(self.kmeansoutput, self.classId, self.features)  ### FEATURES : TRANSPOSE?
        heteClstrFeatureId, heteClstrId = self.trainer.getHeterogenous()
        self.trainer.trainSVM(heteClstrFeatureId,self.subDirectory)


from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
m = M('fan')

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
            #queryjson = request.json
            '''
            text_file = open('./query.json', "w")
            text_file.write(queryjson)
            text_file.close()
            '''
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
    numclass = 3
    numfull = 5
    numpartial = 5
    k = numclass
    print 'MAIN'
    if(doTrain):
        m.trainIt(numclass, numfull, numpartial, k)
        m.saveTraining()
    else:
        m.loadTraining("deneme__CFPK:3_5_5_3")
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    #sess.init_app(app)
    app.debug = True
    app.run(host='0.0.0.0', debug= False)
if __name__ == '__main__':main()