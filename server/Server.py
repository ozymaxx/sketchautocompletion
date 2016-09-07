import sys
import os
for path in os.listdir('..'):
    if not os.path.isfile(os.getcwd() + '/../' + path):
        sys.path.append(os.getcwd() + '/../' + path)
sys.path.append('../../libsvm-3.21/python')
sys.path.append('../../sketchfe/sketchfe')

from extractor import *
from FileIO import *
from Predictor import *
import os
from LibSVM import *
import time
from matplotlib import pyplot as plt
from complexCKMeans import *

from flask import Flask, request, render_template, flash, json

app = Flask(__name__)

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



fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(100, 200)
predictor = None
cudaSupport = None


from sklearn.neighbors import NearestNeighbors
import pickle
dataPath = "../data"
NUM_OF_FULL_SKETCHES = 80

def nn_train():
    trained_nearest = {}

    for imgName in files:
        print("training " + imgName)
        csvpath = dataPath + "/csv/" + imgName + "/" + imgName + ".csv"
        if (os.path.isfile(csvpath)):
            new_bows = predictCSV(csvpath)
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(new_bows)
            trained_nearest[imgName] = neigh


    output = open('ck_nearest.pkl', 'wb')
    pickle.dump(trained_nearest, output)
    output.close()


def load_nn_trainer():
    global nn_dict
    with open('ck_nearest.pkl', 'rb') as f:
        nn_dict = pickle.load(f)


def predictCSV(path):  # extracts the features of all jsons given in path
    #featextractor = IDMFeatureExtractor()
    NUM_OF_FULL_SKETCHES = 80
    NUM_OF_FEATURES = 720
    new_bows = [[0] * NUM_OF_FEATURES] * NUM_OF_FULL_SKETCHES
    with open(path) as f:
        for line in f:
            splitted = line.split(",")
            if (splitted[1] == "False"):
                # new_bows.append(splitted[2:])
                name = splitted[0].split("_")[1].split('.')[0]
                new_bows[int(name) - 1] = splitted[2:]
        f.close()
    return new_bows


def getBestImage_noload(sketch, imgName):
    dataPath = "./sketchautocompletion/data"
    NUM_OF_FULL_SKETCHES = 80
    csvpath = dataPath + "/csv/" + imgName + "/" + imgName + ".csv"
    print(os.path.isfile(csvpath))
    new_bows = predictCSV(csvpath)
    # print(len(new_bows))
    loadedSketch = shapecreator.buildSketch('json', sketch)
    featextractor = IDMFeatureExtractor()
    bows = featextractor.extract(loadedSketch)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(new_bows)
    #print(str(neigh))
    #print(type(neigh))
    output = open('deneme.pkl', 'wb')
    pickle.dump(neigh, output)
    output.close()

    min_dist_index = neigh.kneighbors(bows, return_distance=False)
    print(min_dist_index)

    return min_dist_index[0][0]


def getBestImage(sketch, imgName):
    # print(len(new_bows))
    loadedSketch = shapecreator.buildSketch('json', sketch)
    featextractor = IDMFeatureExtractor()
    bows = featextractor.extract(loadedSketch)
    neigh = nn_dict[imgName]
    min_dist_index = neigh.kneighbors(bows, return_distance=False)
    print(min_dist_index)
    #print(type(min_dist_index))
    #print(type(min_dist_index[0][0]))
    return min_dist_index[0][0] + 1

load_nn_trainer()
#classname = "airplane"
#offset = getBestImage(queryjson, classname)
#print("offset: " + offset)
import base64
#path = "../data/sketches/" + classname + "/" + offset + ".png"
#encoded = base64.b64encode(path)
#print(encoded)

def getBase64(queryjson, serverresponse ):
    print(serverresponse)
    splitted = serverresponse.split("&")
    encodeds = []
    result = ""
    print("Splitted length: " + str(len(splitted)))
    for i in range(0, 5):
        bestImageOffset = getBestImage(queryjson, splitted[i])
        classname = splitted[i]
        bestImageFileName = str(bestImageOffset + files.index(classname)*80)
        fullPathPng = "../data/sketches_downsampled/" + classname + "/" + bestImageFileName + ".png"
        print("Path: " + fullPathPng)
        f = open(fullPathPng, 'rb')
        encodeds.append(base64.b64encode(f.read()))
        f.close()
        result += encodeds[i] + "&"
    return result

def draw_json(jsonString):
    """
    Draws given json in the matplotlib canvas
    To test the server input, if necessary
    """
    xMatrix = []
    yMatrix = []

    if 'CombinedMultiDict' in jsonString:
        jsonString = jsonString[50: len(jsonString) - 30]

    decoded = json.loads(jsonString)

    # print "JSON parsing example: ", decoded['strokes']
    # print "Complex JSON parsing example: ", decoded['two']['list'][1]['item']

    if len(decoded['strokes']):
        for i in range(0, len(decoded['strokes'])):
            xMatrix.append([])
            yMatrix.append([])
            for j in range(0, len(decoded['strokes'][i]['points'])):
                xMatrix[i].append(decoded['strokes'][i]['points'][j]['x'])
                yMatrix[i].append(decoded['strokes'][i]['points'][j]['y'])
    else:
        xMatrix.append([])
        yMatrix.append([])
        xMatrix[0].append(0)
        yMatrix[0].append(0)

    plt.close()
    if len(decoded['strokes']):
        for i in range(0, len(decoded['strokes'])):
            plt.plot(xMatrix[i], yMatrix[i])
    else:
        plt.plot(xMatrix[0], yMatrix[0])

    plt.xlim([0, 1070])
    plt.ylim([0, 752])
    plt.show(block=False)


def classProb2serverResponse(classProb, N, C=0):
    """
    Convert class probabilities -as a dictionary-
    :param classProb: Class Probabilities as a dictionary given by Predictor class
    :param N: Number of guesses
    :param C: threshold, reject instance when probabilities less than the threshold
    :return: Server Response seperated by '&' sign, in the following format: airplane&cat&0.8&0.2
    """
    a = sorted(classProb, key=classProb.get, reverse=True)[:N]
    l = ''
    l1 = ''
    total = 0
    for i in a:
        total += classProb[i]
        l1 += str(classProb[i])
        l += str(files[i])
        l += '&'
        l1 += '&'
    l1 = l1[:-1]

    if total < C:
        return ''

    return l + l1

def classProb2serverResponseBase64(queryjson, classProb, N, C=0):
    """
    Convert class probabilities -as a dictionary-
    :param classProb: Class Probabilities as a dictionary given by Predictor class
    :param N: Number of guesses
    :param C: threshold, reject instance when probabilities less than the threshold
    :return: Server Response seperated by '&' sign, in the following format: airplane&cat&0.8&0.2
    """
    a = sorted(classProb, key=classProb.get, reverse=True)[:N]
    l = ''
    l1 = ''
    total = 0
    for i in a:
        total += classProb[i]
        l1 += str(classProb[i])
        l += str(files[i])
        l += '&'
        l1 += '&'
    l1 = l1[:-1]

    if total < C:
        return ''

    return getBase64(queryjson, l) + l1 + "&" + l[:-1]

def train(trainingName, trainingpath, numclass, numfull, numpartial, k):
    global files
    fio = FileIO()
    extr = Extractor('../data/')
    features, isFull, classId, names, folderList = extr.loadfolders(numclass=numclass,
                                                                    numfull=numfull,
                                                                    numpartial=numpartial)
    global cudaSupport
    if not cudaSupport:
        ckmeans = ComplexCKMeans(features, isFull, classId, k)
        kmeansoutput = ckmeans.getCKMeans()
    else:
        # a late import in order to prevent imporwt errors on computers without cuda support
        from complexcudackmeans import complexCudaCKMeans
        clusterer = complexCudaCKMeans(features, k, classId, isFull)  # FEATURES : N x 720
        clusters, centers = clusterer.cukmeans()
        kmeansoutput = [clusters, centers]

    # find heterogenous clusters and train svm for
    trainer = Trainer(kmeansoutput, classId, features)
    heteClstrFeatureId, heteClstrId = trainer.getHeterogenousClusterId()
    fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                     trainingpath, trainingName)
    svm = trainer.trainSVM(heteClstrFeatureId, trainingpath)
    return kmeansoutput, classId, svm


@app.route("/", methods=['POST', 'GET'])
def handle_data():
    global predictor
    timeStart = time.time()
    try:
        #queryjson = request.args['json']
        queryjson = request.data
        print(queryjson)


        #draw_json(str(queryjson))

        classProb = predictor.predictByString(str(queryjson))
        serverOutput = classProb2serverResponse(classProb, 5)
        print 'Server responded in %.3f seconds' % float(time.time() - timeStart)
        serverOutput = classProb2serverResponseBase64(queryjson, classProb, 5)

        return serverOutput
        return encoded+"&"+encoded+"&"+encoded+"&"+encoded+"&"+encoded+"&"+"1"+"&"+"1"+"&"+"1"+"&"+"1"+"&"+"1"+"&"+"yagmur"+"&"+"yagmur"+"&"+"yagmur"+"&"+"yagmur"+"&"+"yagmur"

    except Exception as e:
        flash(e)
        print(request.values)
        return "Error" + str(e)


@app.route("/send", methods=['POST', 'GET'])
def return_probabilities():
    print 'SERVER: return_probabilities method called'
    raise NotImplementedError


@app.route("/home", methods=['GET'])
def homepage():
    print 'SERVER: homepage method called'
    return render_template("index.html")


def main():
    ForceTrain = False
    numclass, numfull, numpartial = 250, 80, 80
    k = numclass
    training_name = '%s__%i_%i_%i_%i' % ('Main-CUDA__CFPK', numclass, numfull, numpartial, k)
    training_name = 'Main-CUDA__CFPK_250_80_250_250'
    training_path = '../data/training/' + training_name

    # check if pycuda is installed
    global cudaSupport
    import imp
    try:
        imp.find_module('pycuda')
        cudaSupport = True
    except ImportError:
        cudaSupport = False

    # if training data is already computed, import the model
    if os.path.exists(training_path) and not ForceTrain:
        fio = FileIO()
        names, classid, isfull, features, kmeansoutput, myfiles = fio.loadTraining(training_path + "/" + training_name)
        svm = Trainer.loadSvm(kmeansoutput, classid, training_path, features)
        svm.loadModels()
    else:  # otherwise train from data
        kmeansoutput, classid, svm = train(training_name, training_path, numclass, numfull, numpartial, k)

    # form the predictor
    global predictor
    predictor = Predictor(kmeansoutput, classid, training_path, svm=svm)

    # start the server
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', debug=False)
    print 'Server ended'
if __name__ == '__main__': main()
