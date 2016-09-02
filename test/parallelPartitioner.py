"""
Ahmet BAGLAN

Parallel partioner divides data as test and train and saves to testingData and trainingData
"""

import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
sys.path.append('../clusterer/')
sys.path.append('../classifiers/')
sys.path.append('../data/')
sys.path.append("../../libsvm-3.21/python/")

from extractor import *
from featureutil import *
from SVM import *
from Predictor import *
from FileIO import *

debugModeOn = False

def partitionfeatures(features, isFull, classId, names, numtrainfull, saveName = None):

    numclass = len(set(classId))

    # condition for being training data

    cond = [bool(processName(names[index])[1] < numtrainfull) for index in range(len(features))]
    test_features = [features[index] for index in range(len(features)) if not cond[index]]
    test_isFull = [isFull[index] for index in range(len(isFull)) if not cond[index]]
    test_names = [names[index] for index in range(len(names)) if not cond[index]]
    test_classId = [classId[index] for index in range(len(classId)) if not cond[index]]

    # remove any partial sketch from the traning data
    train_features = [features[index] for index in range(len(features)) if cond[index]]
    train_names = [names[index] for index in range(len(names)) if cond[index]]
    train_isFull = [isFull[index] for index in range(len(isFull)) if cond[index]]
    train_classId = [classId[index] for index in range(len(classId)) if cond[index]]

    myfio = FileIO()
    myfio.save(train_isFull, train_names, train_features, '../data/trainingData/csv/' + saveName+'/'+saveName+'.csv')
    myfio.save(test_isFull,test_names,test_features, '../data/testingData/csv/'+ saveName+'/'+saveName+'.csv')


def partition(numclass, numfull, numpartial, numtest):

    print "PARTITION STARTED"
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

    extr = Extractor('../data/')
    for i in range(numclass):
        whole_features, \
        whole_isFull, \
        whole_classId, \
        whole_names, \
        folderList = extr.loadfolders(
                                1,
                                numfull=numfull,
                                numpartial=numpartial,
                                folderList=files[i:])


        saveName = files[i:][0]

        if debugModeOn:
            print type(whole_classId),type(whole_features),type(whole_isFull),type(whole_names)
            print "Doing for partition for ", saveName
            print folderList
        partitionfeatures(whole_features,
                          whole_isFull,
                          whole_classId,
                          whole_names,
                          numtrainfull = numfull-numtest,
                          saveName = saveName)
    print "PARTITION ENDED"
