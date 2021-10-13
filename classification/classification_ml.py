#encoding=utf-8
'''
Created on May 29, 2018

@author: liyuying
'''


import os
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import random
from sklearn.naive_bayes import GaussianNB
from xlutils.copy import copy
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier 

import Text.classifier1 as t_cla
import Image.classifier1 as i_cla
from sklearn import preprocessing
import MySQLdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from xlutils.copy import copy
import xlrd
import xlwt
import os


from sklearn.model_selection import train_test_split
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import keras_metrics

FREQUENCY = 30
appName = ['MyListening', 'HuJiang', 'HuaWei', '10010000000024', 'Game2048', 'SLife']
appID = ['10010000000019', '10010000000037', '10010000000036', '10010000000024', '10010000000017', '10010000000032']

# 调参
# KNN：
N_NEIGHBORS = 10
LEAF_SIZE = 30
WEIGHTS = 'uniform'

def categoryStr(num):
    if num == 0:
        return "安全"
    if num == 1:
        return "安装失败"
    if num == 2:
        return "不正常退出"
    if num == 3:
        return "登录异常"
    if num == 4:
        return "更新异常"
    if num == 5:
        return "功能不完整"
    if num == 6:
        return "性能"
    if num == 7:
        return "页面布局缺陷"
    if num == 8:
        return "用户体验"
    if num == 9:
        return "注册异常"
    if num == 10:
        return "其他"


def getData(appName, appID):
    load_fn = 'Image/mfiles/result/' + appName + '.mat'
    conn = MySQLdb.connect(host='localhost', user='root', passwd='root', port=3306)
    cur = conn.cursor()
    conn.select_db('Kikbug')
    cur.execute("SET character_set_client = utf8;")
    cur.execute("SET character_set_connection = utf8;")
    cur.execute("SET character_set_results = utf8;")
    cur.execute("SET character_set_server = utf8;")

    sql = "select bug_id from training_set where case_id = %s and img_url !=\"\" order by bug_id"
    cur.execute(sql,[appID])
    res = cur.fetchall()
    conn.commit()

    t_train_data, all_train_target = t_cla.NB(appID)
    tmp = np.array(t_train_data)
    normalizer = preprocessing.Normalizer().fit(tmp)
    a = normalizer.transform(tmp)
    t_train_data = a.tolist()


    i_train_data, all_train_target = i_cla.getVector(load_fn, appName, appID)
    tmp = np.array(i_train_data)
    normalizer = preprocessing.Normalizer().fit(tmp)
    a = normalizer.transform(tmp)
    i_train_data = a.tolist()

    flag = 0
    all_train_data = []
    for one in t_train_data:
        all_train_data.append(one + i_train_data[flag])
        flag += 1

    # tmp = np.array(all_train_data)
    # normalizer = preprocessing.Normalizer().fit(tmp)
    # a = normalizer.transform(tmp)
    # all_train_data = a.tolist()
    train_data, test_data, train_target, test_target = train_test_split(all_train_data, all_train_target, test_size=0.1)
    return [train_data, test_data, train_target, test_target]

def classifierALL(appName, appID, row, col, dataset, wb):
    load_fn = 'Image/mfiles/result/' + appName + '.mat'
    train_data = []
    test_data = []
    train_data1 = []

    # ---------------------------- new ----------------------------
    for tmp in dataset[0]:
        train_data.append(tmp[0:4299])

    for tmp in dataset[1]:
        test_data.append(tmp[0:4299])
    # ---------------------------- new -----------------------------

    # for tmp in dataset[0]:
    #     train_data1.append(tmp[0:100])
    # tmp = np.array(train_data1)
    # normalizer = preprocessing.Normalizer().fit(tmp)
    # a = normalizer.transform(tmp)
    # train_data1 = a.tolist()

    # flag = 0
    # for tmp in dataset[0]:
    #     train_data.append(train_data1[flag]+tmp[99:4299])

    # test_data1 = []
    # for tmp in dataset[1]:
    #     test_data1.append(tmp[0:100])

    # tmp = np.array(test_data1)
    # normalizer = preprocessing.Normalizer().fit(tmp)
    # a = normalizer.transform(tmp)
    # test_data1 = a.tolist()

    # flag = 0
    # for tmp in dataset[1]:
    #     test_data.append(test_data1[flag]+tmp[99:4299])

    # ss = StandardScaler()
    # train_data = ss.fit_transform(train_data)
    # test_data = ss.transform(test_data)

    tmp = np.array(train_data)
    normalizer = preprocessing.Normalizer().fit(tmp)
    a = normalizer.transform(tmp)
    train_data = a.tolist()


    tmp = np.array(test_data)
    normalizer = preprocessing.Normalizer().fit(tmp)
    a = normalizer.transform(tmp)
    test_data = a.tolist()

    # test_data = dataset[1]
    train_target = dataset[2]
    test_target = dataset[3]

    # ss = StandardScaler()
    # train_data = ss.fit_transform(train_data)
    # test_data = ss.transform(test_data)

    # tmp = np.array(train_data)
    # normalizer = preprocessing.Normalizer().fit(tmp)
    # a = normalizer.transform(tmp)
    # train_data = a.tolist()

    tmp = np.array(test_data)
    normalizer = preprocessing.Normalizer().fit(tmp)
    a = normalizer.transform(tmp)
    test_data = a.tolist()

    # KNN
    knn = KNeighborsClassifier(n_neighbors = N_NEIGHBORS, leaf_size = LEAF_SIZE, weights = WEIGHTS)
    knn.fit(train_data, train_target)
    result = knn.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('KNN-ALL').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('KNN-ALL').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))

     #GaussianNB
    clf = GaussianNB()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('NB-ALL').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('NB-ALL').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))
    

    #RF
    clf = RandomForestClassifier()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('RF-ALL').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('RF-ALL').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))
    

    # DT
    clf = tree.DecisionTreeClassifier(max_depth = 5)
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('DT-ALL').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('DT-ALL').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))

    # CNN
    # data pre-processing
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(train_data)
    X_test = min_max_scaler.fit_transform(test_data)
    y_train =np_utils.to_categorical(train_target, num_classes=11)
    y_test = np_utils.to_categorical(test_target, num_classes=11)

    input_dim = int(str(X_train[1].shape).replace('(','').replace(',)',''))

    # Another way to build your neural net
    model = Sequential([
        Dense(32, input_dim=input_dim),
        Activation('relu'),
        Dense(11),
        Activation('softmax'),
    ])

    # Another way to define your optimizer
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Another way to train the model
    model.fit(X_train, y_train, batch_size=32)

    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])


    # Evaluate the model with the metrics we defined earlier
    loss, accuracy1, precision1, recall1 = model.evaluate(X_test, y_test)
    f1score1 = 2 * (precision1 * recall1) / (precision1 + recall1)

    wb.get_sheet('CNN-ALL').write(row, col, str(accuracy1))
    wb.get_sheet('CNN-ALL').write(row, col +1 , str(f1score1))
    

    #multi-kernel
    clf = SVC(C=2,kernel='rbf')
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('SVM-ALL').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('SVM-ALL').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))

def classifierTXT(appName, appID, row, col, dataset, wb):
    load_fn = 'Image/mfiles/result/' + appName + '.mat'
    train_data = []
    test_data = []
    for tmp in dataset[0]:
        train_data.append(tmp[0:100])
    for tmp in dataset[1]:
        test_data.append(tmp[0:100])
    train_target = dataset[2]
    test_target = dataset[3]
    
    tmp = np.array(test_data)
    normalizer = preprocessing.Normalizer().fit(tmp)
    a = normalizer.transform(tmp)
    test_data = a.tolist()

    # KNN
    knn = KNeighborsClassifier(n_neighbors = N_NEIGHBORS, leaf_size = LEAF_SIZE, weights = WEIGHTS)
    knn.fit(train_data, train_target)
    result = knn.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('KNN-TXT').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('KNN-TXT').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))

     #GaussianNB
    clf = GaussianNB()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('NB-TXT').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('NB-TXT').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))


    #RF
    clf = RandomForestClassifier()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('RF-TXT').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('RF-TXT').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))
    

    # DT
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('DT-TXT').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('DT-TXT').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))

    # CNN
    # data pre-processing
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(train_data)
    X_test = min_max_scaler.fit_transform(test_data)
    y_train =np_utils.to_categorical(train_target, num_classes=11)
    y_test = np_utils.to_categorical(test_target, num_classes=11)

    input_dim = int(str(X_train[1].shape).replace('(','').replace(',)',''))

    # Another way to build your neural net
    model = Sequential([
        Dense(32, input_dim=input_dim),
        Activation('relu'),
        Dense(11),
        Activation('softmax'),
    ])

    # Another way to define your optimizer
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Another way to train the model
    model.fit(X_train, y_train, batch_size=32)

    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])


    # Evaluate the model with the metrics we defined earlier
    loss, accuracy1, precision1, recall1 = model.evaluate(X_test, y_test)
    f1score1 = 2 * (precision1 * recall1) / (precision1 + recall1)

    wb.get_sheet('CNN-TXT').write(row, col, str(accuracy1))
    wb.get_sheet('CNN-TXT').write(row, col +1 , str(f1score1))
    

    #multi-kernel
    clf = SVC(C=2,kernel='sigmoid')
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('SVM-TXT').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('SVM-TXT').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))

def classifierIMG(appName, appID, row, col, dataset, wb):
    load_fn = 'Image/mfiles/result/' + appName + '.mat'
    train_data = []
    test_data = []
    for tmp in dataset[0]:
        train_data.append(tmp[99:4300])
    for tmp in dataset[1]:
        test_data.append(tmp[99:4300])
    train_target = dataset[2]
    test_target = dataset[3]

    # tmp = np.array(train_data)
    # normalizer = preprocessing.Normalizer().fit(tmp)
    # a = normalizer.transform(tmp)
    # train_data = a.tolist()

    # tmp = np.array(test_data)
    # normalizer = preprocessing.Normalizer().fit(tmp)
    # a = normalizer.transform(tmp)
    # test_data = a.tolist()
    
    tmp = np.array(test_data)
    normalizer = preprocessing.Normalizer().fit(tmp)
    a = normalizer.transform(tmp)
    test_data = a.tolist()

    # KNN
    knn = KNeighborsClassifier(n_neighbors = N_NEIGHBORS, leaf_size = LEAF_SIZE, weights = WEIGHTS)
    knn.fit(train_data, train_target)
    result = knn.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('KNN-IMG').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('KNN-IMG').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))

     #GaussianNB
    clf = GaussianNB()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('NB-IMG').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('NB-IMG').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))
    

    #RF
    clf = RandomForestClassifier()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('RF-IMG').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('RF-IMG').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))
    

    # DT
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('DT-IMG').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('DT-IMG').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))

    # CNN
    # data pre-processing
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(train_data)
    X_test = min_max_scaler.fit_transform(test_data)
    y_train =np_utils.to_categorical(train_target, num_classes=11)
    y_test = np_utils.to_categorical(test_target, num_classes=11)

    input_dim = int(str(X_train[1].shape).replace('(','').replace(',)',''))

    # Another way to build your neural net
    model = Sequential([
        Dense(32, input_dim=input_dim),
        Activation('relu'),
        Dense(11),
        Activation('softmax'),
    ])

    # Another way to define your optimizer
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Another way to train the model
    model.fit(X_train, y_train, batch_size=32)

    # We add metrics to get more results you want to see
    model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])


    # Evaluate the model with the metrics we defined earlier
    loss, accuracy1, precision1, recall1 = model.evaluate(X_test, y_test)
    f1score1 = 2 * (precision1 * recall1) / (precision1 + recall1)

    wb.get_sheet('CNN-IMG').write(row, col, str(accuracy1))
    wb.get_sheet('CNN-IMG').write(row, col +1 , str(f1score1))
    

    #multi-kernel
    clf = SVC(C=2,kernel='sigmoid')
    clf.fit(train_data, train_target)
    result = clf.predict(test_data)
    test_result = result.tolist()
    wb.get_sheet('SVM-IMG').write(row, col, accuracy_score(test_target, test_result))
    wb.get_sheet('SVM-IMG').write(row, col +1 , f1_score(test_target, test_result, average='weighted'))


if __name__ == "__main__":
    rb = xlrd.open_workbook('/Users/liyuying/Desktop/result.xls')
    wb = copy(rb)
    algorithms = ['KNN', 'NB', 'RF', 'DT', 'SVM' ]
    for x in xrange(0, FREQUENCY):
        for app in xrange(0,6):
            dataset = getData(appName[app],appID[app])
            # TXT
            classifierTXT(appName[app], appID[app], x, 2*app, dataset, wb)
            # IMG
            classifierIMG(appName[app], appID[app], x, 2*app, dataset, wb)
            # ALL
            classifierALL(appName[app], appID[app], x, 2*app, dataset, wb)
    wb.save('/Users/liyuying/Desktop/result'+'.xls')






