"""
Ahmet BAGLAN
"""
import sys
sys.path.append("../../sketchfe/sketchfe")
sys.path.append('../predict/')
from FileIO import *
import numpy as np


def saveCenters(doOnlyFulls = True, getDataFrom = '../data/csv/', numClass =10 ** 6, savePath = '../data/newMethodTraining/allCenters.csv'):
    """This file computes centers of classes in the eitz database and saves it to savePath"""

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

    centerSet  = []
    my_names = []
    my_isFull = []
    f = FileIO()

    #Set number of classes to find centers
    if(numClass == None):
        numClass = len(files)
    else:
        numClass = min(numClass, len(files))


    for mfile in files[:numClass]:

        #Get every instance in the given class
        names, isFull, features = f.load(getDataFrom + mfile + '/' + mfile + '.csv')

        #Initialise centers as 0
        nowCenter = np.zeros(len(features[0]))


        if not doOnlyFulls:
            #Use every full and partial instance in the class and sum them
            totalNumOfInstances = len(features)
            for instance in features:
                nowCenter += instance
        else:
            #Use only full instances in the class and sum them
            totalNumOfInstances = 0
            for i in range(len(features)):
                if isFull[i] == True:#Check the prints to see if this works fine (See if the totalNumOfInstances is meaningful
                    nowCenter+=features[i]
                    totalNumOfInstances+=1

        print mfile,'Used ', totalNumOfInstances, 'of instances to find the class center'

        nowCenter = nowCenter/totalNumOfInstances

        centerSet.append(nowCenter)#Add the mean feature vector for this representative
        my_names.append(mfile)#The name of the representative
        my_isFull.append(0)#Just set anything, not necessary for class representative
    f.save(my_isFull,my_names,centerSet,savePath)


if __name__ == '__main__':saveCenters(True, numClass=50)