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
import numpy as np

from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
predictor = None

files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack', 'banana',
         'barn', 'baseball-bat', 'basket', 'bathtub', 'bear-(animal)', 'bed', 'bee', 'beer-mug', 'bell', 'bench',
         'bicycle', 'binoculars', 'blimp', 'book', 'bookshelf', 'boomerang', 'bottle-opener', 'bowl', 'brain', 'bread',
         'bridge', 'bulldozer', 'bus', 'bush', 'butterfly', 'cabinet', 'cactus', 'cake', 'calculator', 'camel', 'camera',
         'candle', 'cannon', 'canoe', 'car-(sedan)', 'carrot', 'castle', 'cat', 'cell-phone', 'chair', 'chandelier',
         'church', 'cigarette', 'cloud', 'comb', 'computer-monitor', 'computer-mouse', 'couch', 'cow', 'crab',
         'crane-(machine)', 'crocodile', 'crown', 'cup', 'diamond', 'dog', 'dolphin', 'donut', 'door', 'door-handle',
         'dragon', 'duck', 'ear', 'elephant', 'envelope', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fire-hydrant',
         'fish', 'flashlight', 'floor-lamp', 'flower-with-stem', 'flying-bird', 'flying-saucer', 'foot', 'fork', 'frog',
         'frying-pan', 'giraffe', 'grapes', 'grenade', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'head',
         'head-phones', 'hedgehog', 'helicopter', 'helmet', 'horse', 'hot-air-balloon', 'hot-dog', 'hourglass', 'house',
         'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'key', 'keyboard', 'knife', 'ladder', 'laptop', 'leaf',
         'lightbulb', 'lighter', 'lion', 'lobster', 'loudspeaker', 'mailbox', 'megaphone', 'mermaid', 'microphone',
         'microscope', 'monkey', 'moon', 'mosquito', 'motorbike', 'mouse-(animal)', 'mouth', 'mug', 'mushroom', 'nose',
         'octopus', 'owl', 'palm-tree', 'panda', 'paper-clip', 'parachute', 'parking-meter', 'parrot', 'pear', 'pen',
         'penguin', 'person-sitting', 'person-walking', 'piano', 'pickup-truck', 'pig', 'pigeon', 'pineapple',
         'pipe-(for-smoking)', 'pizza', 'potted-plant', 'power-outlet', 'present', 'pretzel', 'pumpkin', 'purse',
         'rabbit', 'race-car', 'radio', 'rainbow', 'revolver', 'rifle', 'rollerblades', 'rooster', 'sailboat',
         'santa-claus', 'satellite', 'satellite-dish', 'saxophone', 'scissors', 'scorpion', 'screwdriver', 'sea-turtle',
         'seagull', 'shark', 'sheep', 'ship', 'shoe', 'shovel', 'skateboard']

def getBestPredictions(classProb, n):
    global files
    a = sorted(classProb, key=classProb.get, reverse=True)[:n]
    l = ''
    l1 = ''
    for i in a:
        l1 += str(classProb[i])
        l += files[i]
        l += '&'
        l1 += '&'
    l1 = l1[:-1]
    return l + l1




def newTraining(n,files, numclass, numfull,numpartial,k):
    extr = Extractor('../data/')
    fio = FileIO()

    for i in range(n/5):
        trainingName = '%s_%i__CFPK_%i_%i_%i_%i' % ('training',i, numclass, numfull, numpartial, k)
        trainingpath = '../data/newMethodTraining/' + trainingName
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


@app.route("/", methods=['POST','GET'])
def handle_data():
    global predictor
    timeStart = time.time()
    try:
        queryjson = request.args.get('json')
        classProb = predictor.predictByString(str(queryjson))
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
    numclass, numfull, numpartial = 10, 6, 3
    k = numclass
    n = 10

    # if training data is already computed, import
    if  not ForceTrain:
        pass
        # names, classId, isFull, features, kmeansoutput = fio.loadTraining(trainingpath + "/" + trainingName)
    else:
        newTraining(n , files, numclass, numfull, numpartial, k)
    # global predictor
    # predictor = Predictor(kmeansoutput, classId, trainingpath)
    #
    # app.secret_key = 'super secret key'
    # app.config['SESSION_TYPE'] = 'filesystem'
    # app.debug = True
    # app.run(host='0.0.0.0', debug=False)
    print 'Server ended'
if __name__ == '__main__':main()
