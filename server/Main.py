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

class M:
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
        self.n = 5
    def getBestPredictions(self, c):
        a = sorted(A, key=A.get, reverse=True)[:5]
        l = ''
        for i in a:
            l += self.files[i]
            l += '&'
        l = l[:-1]
        return l


    def trainIt(self, numClass, numFull, numPartial, k):
        print "TRAININDINFAIND"
        self.features, self.isFull, self.classId, self.names = self.extr.loadfolders(numclass = numClass, numfull=numFull, numpartial=numPartial,
                                                                                     folderList=self.files)
        numtestdata = 5
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
        self.trainer.trainSVM(heteClstrFeatureId)

    def predictIt(self, instance):
            # find the probability of given feature to belong any of athe classes
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb)
        return self.getBestPredictions(classProb)

    def predictByPath(self,fullsketchpath):
        instance = featureExtract(fullsketchpath)
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb)
        #return classProb, max(classProb, key=classProb.get)

        return self.getBestPredictions(classProb)
    def predictByString(self, jstring):
        featextractor = IDMFeatureExtractor()
        instance = featextractor.extract(jstring)
        priorClusterProb = self.trainer.computeProb()
        predictor = Predictor(self.kmeansoutput, self.classId)
        classProb = predictor.calculateProb(instance, priorClusterProb)
        return self.getBestPredictions(classProb)

from flask import Flask, request, render_template, flash, jsonify
app = Flask(__name__)
@app.route("/", methods=['POST','GET'])
def handle_data():
    print 'HANDLE DATA'
    try:
        if (request.method == 'POST'):
            #jsonify(data)
            m = M()
            numclass = 2
            numfull = 40
            numpartial = 5
            k = 2

            m.trainIt(numclass,numclass,numfull,k)
            print(str.values)
            return m.predictByString(str.values)
        else:
                        #jsonify(data)
            m = M()
            numclass = 2
            numfull = 60
            numpartial = 5
            k = 2

            m.trainIt(numclass,numclass,numfull,k)

            queryjson = request.args.get("json")
            text_file = open('./query.json', "w")
            text_file.write(queryjson)
            text_file.close()

            answer = m.predictByPath("./query.json")
            return answer

                #"Hello World - you sent me a GET " + str(request.values)
    except Exception as e:
        flash(e)
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


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    #sess.init_app(app)
    app.debug = True
    app.run(host= '0.0.0.0')