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
from SVM import *
import time
from matplotlib import pyplot as plt

from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
predictor = None

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

fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(100, 200)

def drawJson(jsonString):
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

def getBestPredictions(classProb, n):
    a = sorted(classProb, key=classProb.get, reverse=True)[:n]
    l = ''
    l1 = ''
    for i in a:
        l1 += str(classProb[i])
        l += str(files[i])
        l += '&'
        l1 += '&'
    l1 = l1[:-1]
    return l + l1

def train(trainingName, trainingpath, numclass, numfull, numpartial, k):
        global files
        fio = FileIO()
        extr = Extractor('../data/')
        features, isFull, classId, names, folderList = extr.loadfolders(  numclass   = numclass,
                                                                          numfull    = numfull,
                                                                          numpartial = numpartial,
                                                                          folderList = files[:numclass])
        # check if pycuda is installed
        import imp
        try:
            imp.find_module('pycuda')
            found = True
        except ImportError:
            found = False

        if not found:
            constarr = getConstraints(size=len(features), isFull=isFull, classId=classId)
            ckmeans = CKMeans(constarr, np.transpose(features), k)
            kmeansoutput = ckmeans.getCKMeans()
        else:
            from cudackmeans import *
            clusterer = CuCKMeans(features, k, classId, isFull)  # FEATURES : N x 720
            # print 'pycuda', timeit.timeit(lambda: clusterer.cukmeans(data, clusters), number=rounds)
            clusters, centers = clusterer.cukmeans()
            kmeansoutput = [clusters, centers]

        # find heterogenous clusters and train svm
        trainer = Trainer(kmeansoutput, classId, features)
        heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
        fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                         trainingpath, trainingName)
        svm = trainer.trainSVM(heteClstrFeatureId, trainingpath)
        return kmeansoutput, classId, svm

@app.route("/", methods=['POST','GET'])
def handle_data():
    global predictor
    timeStart = time.time()
    try:
        queryjson = request.args.get('json')
        #queryjson = '{"id":"airplane_1_1","strokes":[{"id":"1", "points":[{"pid":"1", "time":1, "x":167.0000,"y":120.4472},{"pid":"2", "time":1, "x":167.0355,"y":120.5136},{"pid":"3", "time":1, "x":167.1388,"y":120.7018},{"pid":"4", "time":1, "x":167.3053,"y":120.9953},{"pid":"5", "time":1, "x":167.5306,"y":121.3775},{"pid":"6", "time":1, "x":167.8099,"y":121.8320},{"pid":"7", "time":1, "x":168.1387,"y":122.3423},{"pid":"8", "time":1, "x":168.5124,"y":122.8917},{"pid":"9", "time":1, "x":168.9265,"y":123.4638},{"pid":"10", "time":1, "x":169.3762,"y":124.0420},{"pid":"11", "time":1, "x":169.8571,"y":124.6099},{"pid":"12", "time":1, "x":170.3645,"y":125.1508},{"pid":"13", "time":1, "x":170.8938,"y":125.6484},{"pid":"14", "time":1, "x":171.4405,"y":126.0860},{"pid":"15", "time":1, "x":172.0000,"y":126.4472},{"pid":"16", "time":1, "x":172.0000,"y":126.4472},{"pid":"17", "time":1, "x":172.8213,"y":126.9022},{"pid":"18", "time":1, "x":173.5842,"y":127.3075},{"pid":"19", "time":1, "x":174.3019,"y":127.6664},{"pid":"20", "time":1, "x":174.9876,"y":127.9819},{"pid":"21", "time":1, "x":175.6547,"y":128.2573},{"pid":"22", "time":1, "x":176.3163,"y":128.4956},{"pid":"23", "time":1, "x":176.9856,"y":128.7000},{"pid":"24", "time":1, "x":177.6759,"y":128.8737},{"pid":"25", "time":1, "x":178.4005,"y":129.0198},{"pid":"26", "time":1, "x":179.1725,"y":129.1415},{"pid":"27", "time":1, "x":180.0053,"y":129.2420},{"pid":"28", "time":1, "x":180.9119,"y":129.3243},{"pid":"29", "time":1, "x":181.9058,"y":129.3916},{"pid":"30", "time":1, "x":183.0000,"y":129.4472},{"pid":"31", "time":1, "x":183.0000,"y":129.4472},{"pid":"32", "time":1, "x":188.5722,"y":129.7135},{"pid":"33", "time":1, "x":193.5796,"y":129.9855},{"pid":"34", "time":1, "x":198.1058,"y":130.2517},{"pid":"35", "time":1, "x":202.2341,"y":130.5002},{"pid":"36", "time":1, "x":206.0481,"y":130.7195},{"pid":"37", "time":1, "x":209.6312,"y":130.8978},{"pid":"38", "time":1, "x":213.0668,"y":131.0234},{"pid":"39", "time":1, "x":216.4385,"y":131.0847},{"pid":"40", "time":1, "x":219.8298,"y":131.0700},{"pid":"41", "time":1, "x":223.3240,"y":130.9676},{"pid":"42", "time":1, "x":227.0046,"y":130.7657},{"pid":"43", "time":1, "x":230.9552,"y":130.4529},{"pid":"44", "time":1, "x":235.2592,"y":130.0172},{"pid":"45", "time":1, "x":240.0000,"y":129.4472},{"pid":"46", "time":1, "x":240.0000,"y":129.4472},{"pid":"47", "time":1, "x":241.1808,"y":129.2523},{"pid":"48", "time":1, "x":242.2569,"y":128.9865},{"pid":"49", "time":1, "x":243.2448,"y":128.6503},{"pid":"50", "time":1, "x":244.1610,"y":128.2443},{"pid":"51", "time":1, "x":245.0218,"y":127.7688},{"pid":"52", "time":1, "x":245.8439,"y":127.2243},{"pid":"53", "time":1, "x":246.6436,"y":126.6115},{"pid":"54", "time":1, "x":247.4375,"y":125.9307},{"pid":"55", "time":1, "x":248.2420,"y":125.1825},{"pid":"56", "time":1, "x":249.0735,"y":124.3673},{"pid":"57", "time":1, "x":249.9486,"y":123.4857},{"pid":"58", "time":1, "x":250.8838,"y":122.5382},{"pid":"59", "time":1, "x":251.8954,"y":121.5251},{"pid":"60", "time":1, "x":253.0000,"y":120.4472},{"pid":"61", "time":1, "x":253.0000,"y":120.4472},{"pid":"62", "time":1, "x":254.3915,"y":119.0606},{"pid":"63", "time":1, "x":255.7380,"y":117.6259},{"pid":"64", "time":1, "x":257.0301,"y":116.1674},{"pid":"65", "time":1, "x":258.2585,"y":114.7091},{"pid":"66", "time":1, "x":259.4141,"y":113.2751},{"pid":"67", "time":1, "x":260.4874,"y":111.8896},{"pid":"68", "time":1, "x":261.4692,"y":110.5767},{"pid":"69", "time":1, "x":262.3502,"y":109.3606},{"pid":"70", "time":1, "x":263.1212,"y":108.2654},{"pid":"71", "time":1, "x":263.7728,"y":107.3152},{"pid":"72", "time":1, "x":264.2958,"y":106.5341},{"pid":"73", "time":1, "x":264.6808,"y":105.9463},{"pid":"74", "time":1, "x":264.9187,"y":105.5760},{"pid":"75", "time":1, "x":265.0000,"y":105.4472},{"pid":"76", "time":1, "x":265.0000,"y":105.4472},{"pid":"77", "time":1, "x":265.0000,"y":105.4472},{"pid":"78", "time":1, "x":265.0000,"y":105.4472},{"pid":"79", "time":1, "x":265.0000,"y":105.4472},{"pid":"80", "time":1, "x":265.0000,"y":105.4472},{"pid":"81", "time":1, "x":265.0000,"y":105.4472},{"pid":"82", "time":1, "x":265.0000,"y":105.4472},{"pid":"83", "time":1, "x":265.0000,"y":105.4472},{"pid":"84", "time":1, "x":265.0000,"y":105.4472},{"pid":"85", "time":1, "x":265.0000,"y":105.4472},{"pid":"86", "time":1, "x":265.0000,"y":105.4472},{"pid":"87", "time":1, "x":265.0000,"y":105.4472},{"pid":"88", "time":1, "x":265.0000,"y":105.4472},{"pid":"89", "time":1, "x":265.0000,"y":105.4472},{"pid":"90", "time":1, "x":265.0000,"y":105.4472}]}]}'
        drawJson(str(queryjson))

        classProb = predictor.predictByString(str(queryjson))
        serverOutput = getBestPredictions(classProb, 5)
        print 'Server responded in %.3f seconds' % float(time.time()-timeStart)
        return serverOutput
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
    ForceTrain = False
    numclass, numfull, numpartial = 5, 6, 3
    k = numclass
    trainingName = '%s__CFPK_%i_%i_%i_%i' % ('Main', numclass, numfull, numpartial, k)
    trainingpath = '../data/training/' + trainingName
    fio = FileIO()

    # if training data is already computed, import
    if os.path.exists(trainingpath) and not ForceTrain:
        names, classId, isFull, features, kmeansoutput, myfiles = fio.loadTraining(trainingpath + "/" + trainingName)
        svm = Trainer.loadSvm(kmeansoutput, classId, trainingpath, features)
    else:
        kmeansoutput, classId, svm = train(trainingName, trainingpath, numclass, numfull, numpartial, k)
    global predictor
    predictor = Predictor(kmeansoutput, classId, trainingpath, svm=svm)

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run(host='0.0.0.0', debug=False)
    print 'Server ended'
if __name__ == '__main__': main()
