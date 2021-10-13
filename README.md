 # Classifying Crowdsourced Mobile Test Reports with Image Features

Abstract Crowdsourced testing has become a popular mobile application testing method, and it is capable of simulating real usage scenarios and detecting various bugs with a large workforce. However, inspecting and classifying the overwhelming number of crowdsourced test reports has become a time-consuming yet inevitable task. To alleviate such tasks, in the past decades, software engineering researchers have proposed many automatic test report classification techniques. However, these techniques may become less effective for crowdsourced mobile application testing, where test reports often consist of insufficient text descriptions and rich screenshots and are fundamentally different from those of traditional desktop software. 

To bridge the gap, we firstly fuse features extracted from text descriptions and screenshots to classify crowdsourced test reports via machine learning and deep learning. We implemented six classification methods of crowdsourced test reports based on machine learning and deep learning:

* Naive Bayes (NB)
* k-Nearest Neighbors (kNN)
* Support Vector Machine (SVM)
* Decision Tree (DT)
* Random Forest (RF)
* Convolutional Neural Network (CNN) 

## Environment
* Python 2.7
* TensorFlow 1.8
* numpy
* scikit-learn
* scipy

## Operation
###### Data preprocessing
For text, the preprocessing process includes word fragmentation, removal of stop words and synonym replacement, dictionary construction, calculate the frequency of the word using TF-IDF and feature extraction. 

```
python /data_preprocessing/text/wordSegmentation.py
python /data_preprocessing/text/getDict.py
python /data_preprocessing/text/getKeyWords.py
python /data_preprocessing/text/getVector.py
```

For screenshots, we only provide the data feature reading function, and the feature extraction method is shown in [^1]. Feature reading:
`python /data_preprocessing/img/getVector.py`

###### Classification
/classification/classification_ml.py is a test report classification code based on machine learning algorithm, which can be controlled to run a single algorithm or run all of it in the `main` function.

Similarly, /classification/classification_dl.py is an implementation code for classifying test reports based on deep learning algorithms. We provide a way to run CNN.
The classification results will be output to the Mysql database.

######  Experimental results analysis
Run `python /calculate_result/calculate_result.py` can calculate the Accuracy, Recall and F-measure. 

Besides, we provide some example data to support method reproduction.

[^1]: Lazebnik, Svetlana, Cordelia Schmid, and Jean Ponce. "Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories." 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06). Vol. 2. IEEE, 2006.


