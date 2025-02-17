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
from ParallelPredictorMaster import *
from ParallelTrainer import *
import time
from centersEitz import *
import grouper

from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
predictor = None

global showFirstN
showFirstN = 5#The number of predictions to be sent

global files
files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe',
             'backpack', 'banana', 'barn', 'baseball-bat', 'basket', 'bathtub', 'bear-(animal)', 'bed',
             'bee', 'beer-mug', 'bell', 'bench', 'bicycle', 'binoculars', 'blimp', 'book', 'bookshelf',
             'boomerang', 'bottle-opener', 'bowl', 'brain', 'bread', 'bridge', 'bulldozer', 'bus', 'bush',
             'butterfly', 'cabinet', 'cactus', 'cake', 'calculator', 'camel', 'camera', 'candle', 'cannon',
             'canoe', 'car-(sedan)', 'carrot', 'castle', 'cat', 'cell-phone', 'chair', 'chandelier',
             'church', 'cigarette', 'cloud', 'comb', 'computer-monitor', 'computer-mouse', 'couch', 'cow',
             'crab', 'crane-(machine)', 'crocodile', 'crown', 'cup', 'diamond', 'dog', 'dolphin', 'donut',
             'door', 'door-handle', 'dragon', 'duck', 'ear', 'elephant', 'envelope', 'eye', 'eyeglasses',
             'face', 'fan', 'feather', 'fire-hydrant', 'fish', 'flashlight', 'floor-lamp', 'flower-with-stem',
             'flying-bird', 'flying-saucer', 'foot', 'fork', 'frog', 'frying-pan', 'giraffe', 'grapes',
             'grenade', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'head', 'head-phones',
             'hedgehog', 'helicopter', 'helmet', 'horse', 'hot-air-balloon', 'hot-dog', 'hourglass',
             'house', 'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'key', 'keyboard', 'knife',
             'ladder', 'laptop', 'leaf', 'lightbulb', 'lighter', 'lion', 'lobster', 'loudspeaker',
             'mailbox', 'megaphone', 'mermaid', 'microphone', 'microscope', 'monkey', 'moon', 'mosquito',
             'motorbike', 'mouse-(animal)', 'mouth', 'mug', 'mushroom', 'nose', 'octopus', 'owl',
             'palm-tree', 'panda', 'paper-clip', 'parachute', 'parking-meter', 'parrot', 'pear', 'pen',
             'penguin', 'person-sitting', 'person-walking', 'piano', 'pickup-truck', 'pig', 'pigeon',
             'pineapple', 'pipe-(for-smoking)', 'pizza', 'potted-plant', 'power-outlet', 'present',
             'pretzel', 'pumpkin', 'purse', 'rabbit', 'race-car', 'radio', 'rainbow', 'revolver',
             'rifle', 'rollerblades', 'rooster', 'sailboat', 'santa-claus', 'satellite', 'satellite-dish',
             'saxophone', 'scissors', 'scorpion', 'screwdriver', 'sea-turtle', 'seagull', 'shark',
             'sheep', 'ship', 'shoe', 'shovel', 'skateboard', 'skull', 'skyscraper', 'snail', 'snake',
             'snowboard', 'snowman', 'socks', 'space-shuttle', 'speed-boat', 'spider', 'sponge-bob',
             'spoon', 'squirrel', 'standing-bird', 'stapler', 'strawberry', 'streetlight', 'submarine',
             'suitcase', 'sun', 'suv', 'swan', 'sword', 'syringe', 't-shirt', 'table', 'tablelamp',
             'teacup', 'teapot', 'teddy-bear', 'telephone', 'tennis-racket', 'tent', 'tiger', 'tire',
             'toilet', 'tomato', 'tooth', 'toothbrush', 'tractor', 'traffic-light', 'train', 'tree',
             'trombone', 'trousers', 'truck', 'trumpet', 'tv', 'umbrella', 'van', 'vase', 'violin',
             'walkie-talkie', 'wheel', 'wheelbarrow', 'windmill', 'wine-bottle', 'wineglass', 'wrist-watch',
             'zebra']

def getBestPredictions(classProb, n):#The function which computes the text to be sent

        a = sorted(classProb, key=classProb.get, reverse=True)[:n]
        l = ''#Text for names
        l1 = ''#Text for probabilites
        for i in a:
            l1 += str(classProb[i])
            l += i
            l += '&'
            l1 += '&'
        l1 = l1[:-1]
        return l + l1

def trainIt(numOfGroups, files, numclass, numfull, numpartial, k, name, doKMeans = True):#The function to be called if train
    #Get groups
    if doKMeans:
        groups = grouper.doKMeansGrouping(numOfGroups,files[:numclass])
    else:
        groups = grouper.doRandomGrouping(numOfGroups,files[:numclass])

    #Get klist (here set every k is set to the number of class representative in group
    klist = grouper.kListGenerator(groups)
    #Train
    myParallelTrainer = ParallelTrainer (name,groups,klist)
    myParallelTrainer.trainSVM(numclass, numfull, numpartial)


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
    ForceTrain = True#Set true if you want to force train
    my_numclass = 7
    my_numfull = 80
    my_numpartial= 80
    my_k = my_numclass
    numOfGroups = 2
    nameOfTheTraining = 'Yejjjkk12445'
    import os
    trainingpath = '../data/newMethodTraining/' + nameOfTheTraining

    # if training data is already computed, import
    if os.path.exists(trainingpath) and (not ForceTrain):
        predictor = ParallelPredictorMaster(nameOfTheTraining)
    else:
        print "CALCULATING CLASS CENTERS"
        saveCenters(doOnlyFulls= True, getDataFrom = "../data/csv/", numClass = my_numclass)
        trainIt(numOfGroups, files[:min(my_numclass,len(files))], my_numclass, my_numfull, my_numpartial, my_k, nameOfTheTraining)
        predictor = ParallelPredictorMaster(nameOfTheTraining)

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run(host='0.0.0.0', debug=False)
    print 'Server ended'
if __name__ == '__main__':main()
