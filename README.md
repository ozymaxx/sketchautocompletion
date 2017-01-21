# Auto-Completion Framework on Sketched Symbols
We implemented an auto-completion system for sketched symbols that is described in [here](http://iui.ku.edu.tr/sezgin_publications/2012/PR%202012%20Sezgin.pdf).

## Dependencies
* **scipy**,
* **numpy**,
* **matplotlib**,
* [sketchfe](https://github.com/ozymaxx/sketchfe),
* **libsvm**,
* **pycuda** if the graphics card supports CUDA parallelization


You can install the first 3 libraries by typing the following in the terminal:
```
sudo apt-get install pip
sudo pip install numpy scipy matplotlib
```
The necessary steps to install the feature extractor can be found in the README file of [this repository](https://github.com/ozymaxx/sketchfe).

We use libSVM as the Support Vector Machine implementation, the necessary information about the installation can be found in [here](https://www.csie.ntu.edu.tw/~cjlin/libsvm/).

## Directory structure
* `clusterer/`: Source code of the constrained K-means clusterer
* `classifiers/`: Source code of SVM classification for every cluster
* `parallelMethod/`: In order to reduce the running time of training, we also parallelized the training pipeline and tested it. This folder contains the source code of that pipeline.
* `SVM/`: Source code of the interface, allowing us to train SVMs on clusters in a parallelized fashion
* `data/`: Source code of the script that extracts the feature of sketches to be trained and saves these features as CSV files.
* `predict/`: Source code of the prediction pipeline
* `server/`: Sketch classification server, which is used in the demonstration application.
* `test/`: Necessary scripts to test the classification performance on both [NicIcon](http://www.ralphniels.nl/pubs/niels-nicicon.pdf) and [Eitz]() data sets

## How to run/use
### Training
The predictor of the system needs a model which is trained on a data set of sketched symbols. To train a model, you should follow these steps:

* Open the terminal
* If the data set only includes the full symbols, extend the data set such that the drawable stroke combinations of the full sketches also exist. These sketches must be in [XML](http://rationale.csail.mit.edu/ETCHASketches/format/) format and be separated into 2 main folders, `test/` for accuracy testing and `train/` for model training. Then, just set `symbolspat` variable to the path to the directory having these two folders, type `cd data` and `python generatepartial.py`. You can ignore this step if you have all the partials.
* Now we have obtained the partial combinations and it's the time for feature extraction. Again, just set `symbolspat` variable to the directory including the sketch set. Then extract the features of them by typing `python savecsv.py`.
* Cluster the sketch instances on the feature representations and construct an SVM model for every cluster. There are many alternative constrained clusterers:
  * `ckmeans.py`: The naive constrained K-means implementation, which can be found in [here](http://nichol.as/papers/Wagstaff/Constrained%20k-means%20clustering%20with%20background.pdf).
  * `cudackmeans.py`: CUDA-accelerated version of `ckmeans.py`.
  * `complexCKmeans.py`: Constrained K-means with voting. The details of this algorithm can be found in this technical report (see section 3.1.3 of [this report](http://iui.ku.edu.tr/KUSRP/KUSRP_IUI_16.pdf)).
  * `complexcudackmeans.py`: CUDA-accelerated version of `complexCKmeans.py`.
  The tests showed that the running time of the constrained K-means algorithm is less than the naive one. Therefore our suggestion is to use the complex algorithm. Moreover, if the graphics driver of your machine supports CUDA, `complexcudackmeans.py` will be far better for you. Our trainer function named `train()` checks whether your machine has CUDA, and then calls the clusterer function accordingly. Although we included the naive implementation, we preferred to call the complex method as the default clusterer method.
  
  In order to call `train()`, just execute the following lines to check the availability of CUDA first:
  ```
  # check if pycuda is installed
  global cudaSupport
  import imp
  try:
      imp.find_module('pycuda')
      cudaSupport = True
  except ImportError:
      cudaSupport = False
  ```
  
  Then, call `train()` function in the following way. This function also saves the trained model.
  ```
  training_name = [name of the model]
  training_path = [path to the model which will be created]
  numclass = [number of classes]
  numfull = [number of full instances in train/ folder]
  numpartial = [number of partial instances in train/ folder]
  k = [number of clusters that will be created]
  kmeansoutput, classid, svm = train(training_name, training_path, numclass, numfull, numpartial, k)
  ```
  
  Once you've done all these steps, the model will be ready.
  
### Prediction
To run the prediction pipeline, the first step is obtaining the JSON representation of the sketched input (more information can be found in [this README document](https://github.com/ozymaxx/sketchfe)). How you create and send this string is up to the implementation of the client.

Before processing the input, firstly load the model into the memory:
```
# form the predictor
predictor = Predictor(kmeansoutput, classid, training_path, svm=svm)
```

Once you've obtained the JSON representation of the sketched input, you will classify it in the following way:
```
draw_json(str(queryjson))

classProb = predictor.predictByString(str(queryjson))
serverOutput = classProb2serverResponse(classProb, 5)
```
where `draw_json()` and `classProb2serverResponse()` can be found in `server/Server.py` and `serverOutput` includes the classification results (most probable classes with their probabilities) as a JSON string.

## Alternative implementation of the training/prediction pipeline
The details of this pipeline can be found in `parallelMethod/` directory. The idea of this implementation can be found in section 3.1.9 of [this report](http://iui.ku.edu.tr/KUSRP/KUSRP_IUI_16.pdf). We just tried this technique out and it gave as the same results as the original pipeline.

## Contact
Ozan Can Altıok - [Koç University IUI Laboratory](http://iui.ku.edu.tr) - oaltiok15 at ku dot edu dot tr<br>
Ahmet Bağlan - Boğaziçi University - ahmet.baglan at boun dot edu dot tr<br>
Arda İçmez - Galatasaray University - aicmez at gmail dot com<br>
Semih Günel - Bilkent University - gunelsemih at gmail dot com<br>
