"""
Ahmet BAGLAN

Server using parallel method

"""

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
from ParallelPredictorSlave import *
from ParallelPredictorMaster import *
from ParallelTrainer import *
import time
from centers import *

from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
predictor = None

global showFirstN
showFirstN = 5
global files
files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack', 'banana',
         'barn', 'baseball-bat', 'basket', 'bathtub', 'bear-(animal)', 'bed', 'bee', 'beer-mug', 'bell', 'bench',
         'bicycle', 'binoculars', 'blimp', 'book', 'bookshelf', 'boomerang', 'bottle-opener', 'bowl', 'brain', 'bread',
         'bridge', 'bulldozer', 'bus', 'bush', 'butterfly', 'cabinet', 'cactus', 'cake', 'calculator', 'camel',
         'camera', 'candle', 'cannon', 'canoe', 'car-(sedan)', 'carrot', 'castle', 'cat', 'cell-phone', 'chair', 'chandelier',
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

def trainIt(n, files, numclass, numfull, numpartial, k, name, doKMeans = True):
    myParallelTrainer = ParallelTrainer (n,files, doKMeans = doKMeans)
    myParallelTrainer.trainSVM(numclass, numfull, numpartial, k, name)


@app.route("/", methods=['POST','GET'])
def handle_data():
    global predictor,my_numclass,my_k,my_numfull,my_numpartial
    timeStart = time.time()
    try:
        #Get the json file
        queryjson = request.data
        #Get predicted dict
        classProb = predictor.predictByString(queryjson)
        print 'Server responded in %.3f seconds' % float(time.time()-timeStart)

        #Send back first N prediction
        return getBestPredictions(classProb, showFirstN)
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

    global predictor
    ForceTrain = False
    my_numclass = 20
    my_numfull = 20
    my_numpartial= 20
    my_k = my_numclass
    groupByN = 4
    nameOfTheTraining = 'Yejjjkk12'
    import os
    trainingpath = '../data/newMethodTraining/' + nameOfTheTraining

    # if training data is already computed, import
    if os.path.exists(trainingpath) and (not ForceTrain):
        predictor = ParallelPredictorMaster(nameOfTheTraining)
    else:
        print "CALCULATING CLASS CENTERS"
        saveCenters(doOnlyFulls= True, getDataFrom = "../data/csv/", numClass = my_numclass)
        trainIt(groupByN, files[:min(my_numclass,len(files))], my_numclass, my_numfull, my_numpartial, my_k, nameOfTheTraining)
        predictor = ParallelPredictorMaster(nameOfTheTraining)

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run(host='0.0.0.0', debug=False)
    print 'Server ended'
if __name__ == '__main__':main()
