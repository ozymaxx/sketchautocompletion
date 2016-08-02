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

from flask import Flask, request, render_template, flash, json
app = Flask(__name__)
predictor = None

files = ['airplane','cat','hamburger','paper-clip','alarm-clock','cell-phone','hammer','parachute','angel','chair','hand',
         'parking-meter','ant','chandelier','harp','parrot','apple','church','hat','pear','arm','cigarette','head','pen',
         'armchair','cloud','head-phones','penguin','ashtray','comb','hedgehog','person-sitting','axe','computer-monitor',
         'helicopter','person-walking','backpack','computer-mouse','helmet','piano','banana','couch','horse','pickup-truck',
         'barn','cow','hot-air-balloon','pig','baseball-bat','crab','hot-dog','pigeon','basket','crane-(machine)','hourglass',
         'pineapple','bathtub','crocodile','house','pipe-(for-smoking)','bear-(animal)','crown','human-skeleton','pizza','bed',
         'cup','ice-cream-cone','potted-plant','bee','diamond','ipod','power-outlet','beer-mug','dog','kangaroo','present','bell',
         'dolphin','key','pretzel','bench','donut','keyboard','pumpkin','bicycle','door','knife','purse','binoculars','door-handle',
         'ladder','rabbit','blimp','dragon','laptop','race-car','book','duck','leaf','radio','bookshelf','ear','lightbulb','rainbow',
         'boomerang','elephant','lighter','revolver','bottle-opener','envelope','lion','rifle','bowl','eye','lobster','rollerblades',
         'brain','eyeglasses','loudspeaker','rooster','bread','face','mailbox','sailboat','bridge','fan','megaphone','santa-claus',
         'bulldozer','feather','mermaid','satellite','bus','fire-hydrant','microphone','satellite-dish','bush','fish','microscope',
         'saxophone','butterfly','flashlight','monkey','scissors','cabinet','floor-lamp','moon','scorpion','cactus','flower-with-stem',
         'mosquito','screwdriver','cake','flying-bird','motorbike','seagull','calculator','flying-saucer','mouse-(animal)','sea-turtle',
         'camel','foot','mouth','shark','camera','fork','mug','sheep','candle','frog','mushroom','ship','cannon','frying-pan','nose',
         'shoe','canoe','giraffe','octopus','shovel','carrot','grapes','owl','skateboard','car-(sedan)','grenade','palm-tree','castle',
         'guitar','panda']

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
    doTrain = True
    numclass, numfull, numpartial = 3, 5, 3
    k = numclass
    trainingName = '%s__CFPK_%i_%i_%i_%i' % ('training', numclass, numfull, numpartial, k)
    trainingpath = '../data/training/' + trainingName
    fio = FileIO()

    # if training data is already computed, import
    if os.path.exists(trainingpath) and not doTrain:
        names, classId, isFull, features, kmeansoutput = fio.loadTraining(trainingpath + "/" + trainingName)
    else:
        extr = Extractor('../data/')
        features, isFull, classId, names, folderList = extr.loadfolders(  numclass   = numclass,
                                                                          numfull    = numfull,
                                                                          numpartial = numpartial,
                                                                          folderList = files )
        constarr = getConstraints(size=len(features), isFull=isFull, classId=classId)
        ckmeans = CKMeans(constarr, np.transpose(features), k)
        kmeansoutput = ckmeans.getCKMeans()

        # find heterogenous clusters and train svm
        trainer = Trainer(kmeansoutput, classId, features)
        heteClstrFeatureId, heteClstrId = trainer.getHeterogenous()
        trainer.trainSVM(heteClstrFeatureId, trainingpath)

        fio.saveTraining(names, classId, isFull, features, kmeansoutput,
                         trainingpath,  trainingName)

    global predictor
    predictor = Predictor(kmeansoutput, classId, trainingpath)

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run(host='0.0.0.0', debug=False)
    print 'Server started'
if __name__ == '__main__':main()
