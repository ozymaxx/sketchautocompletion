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
from my_featutil import *
from Predictor import *

def partition(numclass, numfull, numpartial):

    print "PARTITION STARTED"
    files = ['airplane', 'alarm-clock', 'angel', 'ant', 'apple', 'arm', 'armchair', 'ashtray', 'axe', 'backpack',
             'banana',
             'barn', 'baseball-bat', 'basket', 'bathtub', 'bear-(animal)', 'bed', 'bee', 'beer-mug', 'bell', 'bench',
             'bicycle', 'binoculars', 'blimp', 'book', 'bookshelf', 'boomerang', 'bottle-opener', 'bowl', 'brain',
             'bread',
             'bridge', 'bulldozer', 'bus', 'bush', 'butterfly', 'cabinet', 'cactus', 'cake', 'calculator', 'camel',
             'camera', 'candle', 'cannon', 'canoe', 'car-(sedan)', 'carrot', 'castle', 'cat', 'cell-phone', 'chair',
             'chandelier',
             'church', 'cigarette', 'cloud', 'comb', 'computer-monitor', 'computer-mouse', 'couch', 'cow', 'crab',
             'crane-(machine)', 'crocodile', 'crown', 'cup', 'diamond', 'dog', 'dolphin', 'donut', 'door',
             'door-handle',
             'dragon', 'duck', 'ear', 'elephant', 'envelope', 'eye', 'eyeglasses', 'face', 'fan', 'feather',
             'fire-hydrant',
             'fish', 'flashlight', 'floor-lamp', 'flower-with-stem', 'flying-bird', 'flying-saucer', 'foot', 'fork',
             'frog',
             'frying-pan', 'giraffe', 'grapes', 'grenade', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat',
             'head',
             'head-phones', 'hedgehog', 'helicopter', 'helmet', 'horse', 'hot-air-balloon', 'hot-dog', 'hourglass',
             'house',
             'human-skeleton', 'ice-cream-cone', 'ipod', 'kangaroo', 'key', 'keyboard', 'knife', 'ladder', 'laptop',
             'leaf',
             'lightbulb', 'lighter', 'lion', 'lobster', 'loudspeaker', 'mailbox', 'megaphone', 'mermaid', 'microphone',
             'microscope', 'monkey', 'moon', 'mosquito', 'motorbike', 'mouse-(animal)', 'mouth', 'mug', 'mushroom',
             'nose',
             'octopus', 'owl', 'palm-tree', 'panda', 'paper-clip', 'parachute', 'parking-meter', 'parrot', 'pear',
             'pen',
             'penguin', 'person-sitting', 'person-walking', 'piano', 'pickup-truck', 'pig', 'pigeon', 'pineapple',
             'pipe-(for-smoking)', 'pizza', 'potted-plant', 'power-outlet', 'present', 'pretzel', 'pumpkin', 'purse',
             'rabbit', 'race-car', 'radio', 'rainbow', 'revolver', 'rifle', 'rollerblades', 'rooster', 'sailboat',
             'santa-claus', 'satellite', 'satellite-dish', 'saxophone', 'scissors', 'scorpion', 'screwdriver',
             'sea-turtle',
             'seagull', 'shark', 'sheep', 'ship', 'shoe', 'shovel', 'skateboard']

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

        numtest = 5
        saveName = files[i:][0]
        print "doing for ", saveName
        train_features, train_isFull, train_classId, train_names, test_features, test_isFull, test_names, test_classId = \
            partitionfeatures(whole_features,
                              whole_isFull,
                              whole_classId,
                              whole_names,
                              numtrainfull = numfull-numtest,
                              selectTestRandom=True, saveName = saveName)
    print "PARTITION ENDED"

if __name__ == "__main__": main()