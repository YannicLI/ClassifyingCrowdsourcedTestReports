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
from xlrd import open_workbook
from xlutils.copy import copy
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier 

import Text.classifier1 as t_cla
import Image.classifier1 as i_cla
from sklearn import preprocessing
# import pymysql
# pymysql.install_as_MySQLdb()
import MySQLdb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import datetime
# from sklearn.preprocessing import Imputer
# from sklearn.grid_search import GridSearchCV

FREQUENCY = 30

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


def classifier(load_fn, appName, appID, row, col, cell, alg, xxx, ws):
    ini_col = col
    conn = MySQLdb.connect(host='127.0.0.1', user='root', passwd='root', port=3306)
    cur = conn.cursor()
    conn.select_db('kikbug')
    cur.execute("SET character_set_client = UTF8MB4;")
    cur.execute("SET character_set_connection = UTF8MB4;")
    cur.execute("SET character_set_results = UTF8MB4;")
    cur.execute("SET character_set_server = UTF8MB4;")

    sql = "select bug_id from training_set where case_id = %s and img_url !=\"\" order by bug_id"
    cur.execute(sql,[appID])
    res = cur.fetchall()
    conn.commit()

    #Text:
    if cell != 1:
        t_train_data, all_train_target = t_cla.NB(appID)
        print t_train_data
        tmp = np.array(t_train_data)
        normalizer = preprocessing.Normalizer().fit(tmp)
        a = normalizer.transform(tmp)
        train_data = a.tolist()

        # # 正则化
        # tmp = np.array(t_train_data)
        # X_scaled = preprocessing.minmax_scale(tmp)
        # imp = Imputer(missing_values=0, strategy='mean', verbose=0)
        # imp.fit(X_scaled)
        # b = imp.transform(X_scaled)
        # normalizer = preprocessing.Normalizer().fit(b)
        # a = normalizer.transform(b)
        # t_train_data = a.tolist()

        tmp = np.array(t_train_data)
        normalizer = preprocessing.Normalizer().fit(tmp)
        a = normalizer.transform(tmp)
        t_train_data = a.tolist()


    #Image:
    if cell != 0:
        i_train_data, all_train_target = i_cla.getVector(load_fn, appName, appID)

        # # 正则化
        # tmp = np.array(i_train_data)
        # X_scaled = preprocessing.minmax_scale(tmp)
        # imp = Imputer(missing_values=0, strategy='mean', verbose=0)
        # imp.fit(X_scaled)
        # b = imp.transform(X_scaled)
        # normalizer = preprocessing.Normalizer().fit(b)
        # a = normalizer.transform(b)
        # i_train_data = a.tolist()

        tmp = np.array(i_train_data)
        normalizer = preprocessing.Normalizer().fit(tmp)
        a = normalizer.transform(tmp)
        i_train_data = a.tolist()

    # all
    if cell == 2:
        flag = 0
        all_train_data = []
        for one in t_train_data:
            all_train_data.append(one + i_train_data[flag])
            flag += 1

    # # text
    if cell == 0:
        all_train_data = []
        all_train_data = t_train_data

    # # img
    if cell == 1:
        all_train_data = []
        all_train_data = i_train_data

    tmp = np.array(all_train_data)
    # normalizer = preprocessing.Normalizer().fit(tmp)
    # a = normalizer.transform(tmp)
    all_train_data = a.tolist()

    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0

    for i in range(0, FREQUENCY):
        # train_data = []
        # test_data = []
        # train_target = []
        # test_target = []
        # test_data = train[i]
        # test_target = target[i]
        #
        # for j in range(0, 10):
        #     if j != i:
        #         train_data = train_data + train[j]
        #         train_target = train_target + target[j]


        train_data, test_data, train_target, test_target = train_test_split(all_train_data, all_train_target, test_size=xxx/10.0)


        # scaler = preprocessing.StandardScaler().fit(train_data)
        # scaler.transform(train_data)
        # scaler.transform(test_data)
        #
        # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(train_data)
        # scaler.transform(train_data)
        # scaler.transform(test_data)


        # # KNN
        if alg == 0:
            
        #     # knn = KNeighborsClassifier()
        #     # k_range = list(range(1,10))
        #     # leaf_range = list(range(1,2))
        #     # weight_options = ['uniform','distance']
        #     # algorithm_options = ['auto','ball_tree','kd_tree','brute']
        #     # param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options,leaf_size=leaf_range)
        #     # gridKNN = GridSearchCV(knn,param_gridknn,cv=10,scoring='accuracy',verbose=1)
        #     # gridKNN.fit(train_data,train_target)
        #     # print('best score is:',str(gridKNN.best_score_))
        #     # print('best params are:',str(gridKNN.best_params_))
            start = datetime.datetime.now()
            knn = KNeighborsClassifier()
            end = datetime.datetime.now()
            print "KNN1:"
            print end-start
            start = datetime.datetime.now()
            knn.fit(train_data, train_target)
            end = datetime.datetime.now()
            print "KNN2:"
            print end-start
            start = datetime.datetime.now()
            result = knn.predict(test_data)
            end = datetime.datetime.now()
            print "KNN3:"
            print end-start

         #GaussianNB
        if alg == 1:
            # clf = GaussianNB(alpha=0.1)
            start = datetime.datetime.now()
            clf = GaussianNB()
            end = datetime.datetime.now()
            print "NB1:"
            print end-start
            start = datetime.datetime.now()
            clf.fit(train_data, train_target)
            end = datetime.datetime.now()
            print "NB2:"
            print end-start
            start = datetime.datetime.now()
            result = clf.predict(test_data)
            end = datetime.datetime.now()
            print "NB3:"
            print end-start
            
        

        # #MultinomialNB
        # clf = MultinomialNB()
        # clf.fit(train_data, train_target)
        # result = clf.predict(test_data)

        # #SVM
        # clf = SVC()
        # clf.fit(train_data, train_target)
        # result = clf.predict(test_data)

         #RF
        if alg == 2:
            start = datetime.datetime.now()
            clf = RandomForestClassifier()
            end = datetime.datetime.now()
            print "RF1:"
            print end-start
            start = datetime.datetime.now()
            clf.fit(train_data, train_target)
            end = datetime.datetime.now()
            print "RF2:"
            print end-start
            start = datetime.datetime.now()
            result = clf.predict(test_data)
            end = datetime.datetime.now()
            print "RF3:"
            print end-start


        # DT
        if alg == 3:
            start = datetime.datetime.now()
            clf = tree.DecisionTreeClassifier(max_depth = 5)
            end = datetime.datetime.now()
            print "DT1:"
            print end-start
            start = datetime.datetime.now()
            clf.fit(train_data, train_target)
            end = datetime.datetime.now()
            print "DT2:"
            print end-start
            start = datetime.datetime.now()
            result = clf.predict(test_data)
            end = datetime.datetime.now()
            print "DT3:"
            print end-start

       

        # # LogisticRegression
        # clf = LogisticRegression()
        # clf.fit(train_data, train_target)
        # result = clf.predict(test_data)

        #multi-kernel
        if alg == 4:
            start = datetime.datetime.now()
            clf = SVC(C=2,kernel='sigmoid')
            end = datetime.datetime.now()
            print "SVM1:"
            print end-start
            start = datetime.datetime.now()
            clf.fit(train_data, train_target)
            end = datetime.datetime.now()
            print "SVM2:"
            print end-start
            start = datetime.datetime.now()
            result = clf.predict(test_data)
            end = datetime.datetime.now()
            print "SVM3:"
            print end-start


        # # LinearDiscriminantAnalysis
        # clf = LinearDiscriminantAnalysis()
        # clf.fit(train_data, train_target)
        # result = clf.predict(test_data)

        # # SVM
        # if alg == 5:
        #     svm = SVC(kernel='linear', C=2, gamma=0.2, degree=3, coef0=0.0, shrinking=True, probability=False,
        #                tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo',
        #                random_state=None)
        #     svm.fit(train_data, train_target)
        #     end = datetime.datetime.now()
        #     print "SVM1:"
        #     print end-begin
        #     result = svm.predict(test_data)
        #     end = datetime.datetime.now()
        #     print "SVM2:"
        #     print end-begin

        # LogisticRegression
        if alg == 5:
            start = datetime.datetime.now()
            clf = LogisticRegression()
            end = datetime.datetime.now()
            print "LG1:"
            print end-start
            start = datetime.datetime.now()
            clf.fit(train_data, train_target)
            end = datetime.datetime.now()
            print "LG2:"
            print end-start
            start = datetime.datetime.now()
            result = clf.predict(test_data)
            end = datetime.datetime.now()
            print "LG2:"
            print end-start

        test_result = result.tolist()

        # print accuracy_score(test_target, train_result)
        # print "Accuracy:"
        # print accuracy_score(test_target, train_result)
        accuracy = accuracy + accuracy_score(test_target, test_result)
        ws.write(row, col, accuracy_score(test_target, test_result))
        # print row
        # print col
        col = col + 1
        # # print "Precision:"
        # # print precision_score(test_target, train_result, average='weighted')
        precision = precision + precision_score(test_target, test_result, average='weighted')
        # print 'pre:' + str(precision_score(test_target, test_result, average='weighted'))
        # print precision_score(test_target, test_result, average='weighted')
        # print row
        # print col
        # ws.write(row, col, precision_score(test_target, test_result, average='weighted'))
        # col = col + 1
        # # print "Recall:"
        # # print recall_score(test_target, train_result, average='weighted')
        recall = recall + recall_score(test_target, test_result, average='weighted')
        # print 'rec:' + str(recall_score(test_target, test_result, average='weighted'))
        # print row
        # print col
        # ws.write(row, col, recall_score(test_target, test_result, average='weighted'))
        # col = col + 1
        # # print "f1:"
        # # print f1_score(test_target, train_result, average='weighted')
        f1 = f1 + f1_score(test_target, test_result, average='weighted')
        # print 'f1:' + str(f1_score(test_target, test_result, average='weighted'))
        # print row
        # print col
        ws.write(row, col, f1_score(test_target, test_result, average='weighted'))
        col = ini_col
        row += 1

    
    col = ini_col
    ws.write(FREQUENCY + 2, ini_col, accuracy / FREQUENCY)
    # ws.write(FREQUENCY + 2, ini_col + 1, precision / FREQUENCY)
    # ws.write(FREQUENCY + 2, ini_col + 2, recall / FREQUENCY)
    # ws.write(FREQUENCY + 2, ini_col + 3, f1 / FREQUENCY)
    ws.write(FREQUENCY + 2, ini_col + 1, f1 / FREQUENCY)
    # print "Accuracy:"
    # print accuracy / FREQUENCY
    # #
    # print "Precision:"
    # print precision / FREQUENCY
    # #
    # print "Recall:"
    # print recall / FREQUENCY
    # #
    # print "f1:"
    # print f1 / FREQUENCY
    return

    



if __name__ == "__main__":
    # for i in range(0,11):
    # appName = 'Game2048'
    # load_fn = 'Image/mfiles/result/' + appName + '.mat'
    # appID = "10010000000017"
    # classifier(load_fn, appName, appID)

    # appName = ['MyListening', 'HuJiang', 'HuaWei', '10010000000024', 'Game2048', 'Slife']
    # appID = ['10010000000019', '10010000000037', '10010000000036', '10010000000024', '10010000000017', '10010000000032']

    # appName = ['MyListening','HuaWei']
    # appID = ['10010000000019',  '10010000000036']

    for x in range(1,2):
        rb = open_workbook('E:\\Project\\classification\\classification-result\\result'+str(x)+'.xls', 'w+')
        rs = rb.sheet_by_index(0)
        wb = copy(rb)
        # ws = wb.get_sheet(0)
        row = 2
        col = 1
        # ws.write(FREQUENCY + 2, 0, 'average')

        # appName = ['MyListening', 'HuJiang', 'HuaWei', '10010000000024', 'Game2048', 'SLife']
        # appID = ['10010000000019', '10010000000037', '10010000000036', '10010000000024', '10010000000017', '10010000000032']
        appName = ['HuJiang', 'HuaWei', '10010000000024', 'Game2048', 'SLife']
        appID = ['10010000000037', '10010000000036', '10010000000024', '10010000000017', '10010000000032']
        # appName = ['HuJiang']
        # appID = ['10010000000037']

        # #测全部
        algorithms = ['KNN', 'NB', 'RF', 'DT', 'SVM', 'LG' ]
        # 测某方法：
        # algorithms = ['LG']


        cells = [0]
        # cells = ['TXT']
        # for i in range(2, FREQUENCY + 2):
        #     ws.write(i, 0, 'multi-kernel')
        for cell in cells:
            for alg in range(5,6):
                    # for cell in xrange(2,3):
                    row = 2
                    col = 0
                    print (str(algorithms[alg]))
                    # ws = wb.add_sheet(str(algorithms[alg]) + "-" + str(cells[cell]) + "-" + str(x))
                    if cell == 0:
                        ws = wb.add_sheet(str(algorithms[alg])+'TXT')
                    if cell == 1:
                        ws = wb.add_sheet(str(algorithms[alg])+'IMG')
                    if cell == 2:
                        ws = wb.add_sheet(str(algorithms[alg])+'ALL')
                    for i in range(0, 6):
                        row = 2
                        col = 2 * i + 1
                        load_fn = 'E://Project/classification/ClassifyingCrowdsourcedTestReports/ClassifyingCrowdsourcedTestReports/Image/mfiles/result/' + appName[i] + '.mat'
                        classifier(load_fn, appName[i], appID[i], row, col, cell, alg, x, ws)
        # for alg in xrange(3,4):
        #     for cell in xrange(2,3):
        #         row = 2
        #         col = 0
        #         print str(algorithms[alg])
        #         # ws = wb.add_sheet(str(algorithms[alg]) + "-" + str(cells[cell]) + "-" + str(x))
        #         ws = wb.add_sheet(str(algorithms[alg])+'---ALL-00')
        #         for i in range(0, 6):
        #             row = 2
        #             col = 4 * i + 1
        #             load_fn = 'E://Project/classification/ClassifyingCrowdsourcedTestReports/ClassifyingCrowdsourcedTestReports/Image/mfiles/result/' + appName[i] + '.mat'
        #             # print appName[i]
        #             classifier(load_fn, appName[i], appID[i], row, col, cell, alg, x)
        # wb.save('E:\\Project\\classification\\classification-result\\result'+str(x)+'.xls')






