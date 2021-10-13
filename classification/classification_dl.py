#encoding=utf-8
'''
Created on Mar 29, 2018

@author: liyuying
'''


import os
from sklearn.preprocessing import Imputer
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop



import Text.classifier1 as t_cla
import Image.classifier1 as i_cla
from sklearn import preprocessing
import MySQLdb
from sklearn.model_selection import train_test_split
from xlrd import open_workbook
from xlutils.copy import copy


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


def classifier(appName, appID, row, col):

    conn = MySQLdb.connect(host='localhost', user='root', passwd='root', port=3306)
    cur = conn.cursor()
    conn.select_db('Kikbug')
    cur.execute("SET character_set_client = utf8;")
    cur.execute("SET character_set_connection = utf8;")
    cur.execute("SET character_set_results = utf8;")
    cur.execute("SET character_set_server = utf8;")

    for num in range(0, 6):
        load_fn = 'Image/mfiles/result/' + appName[num] + '.mat'
        sql = "select bug_id from training_set where case_id = %s and img_url !=\"\" order by bug_id"
        cur.execute(sql, [appID[num]])
        res = cur.fetchall()
        conn.commit()

        #Text:
        t_train_data, all_train_target = t_cla.NB(appID[num])

        #Image:
        i_train_data, i_train_target = i_cla.getVector(load_fn, appName[num], appID[num])

        # 正则化


        # tmp = np.array(t_train_data)
        # normalizer = preprocessing.Normalizer().fit(tmp)
        # a = normalizer.transform(tmp)
        # t_train_data = a.tolist()

        # tmp = np.array(i_train_data)
        # normalizer = preprocessing.Normalizer().fit(tmp)
        # a = normalizer.transform(tmp)
        # i_train_data = a.tolist()


        tmp = np.array(t_train_data)
        X_scaled = preprocessing.scale(tmp)
        imp = Imputer(missing_values=0, strategy='mean', verbose=0)
        imp.fit(X_scaled)
        b = imp.transform(X_scaled)

        normalizer = preprocessing.Normalizer().fit(b)
        a = normalizer.transform(b)
        t_train_data = a.tolist()


        tmp = np.array(i_train_data)
        X_scaled = preprocessing.scale(tmp)
        imp = Imputer(missing_values=0, strategy='mean', verbose=0)
        imp.fit(X_scaled)
        b = imp.transform(X_scaled)

        normalizer = preprocessing.Normalizer().fit(b)
        a = normalizer.transform(b)
        i_train_data = a.tolist()

        # # all
        # flag = 0
        # all_train_data = []
        # for one in t_train_data:
        #     all_train_data.append(one + i_train_data[flag])
        #     flag += 1


        # text
        flag = 0
        all_train_data = []
        for one in t_train_data:
            all_train_data.append(one)
            flag += 1

        # # img
        # flag = 0
        # all_train_data = []
        # for one in i_train_data:
        #     all_train_data.append(one)
        #     flag += 1

        train_data, test_data, train_target, test_target = train_test_split(all_train_data, np.array(all_train_target), test_size=0.1, random_state=0)

        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(train_data)
        X_test = min_max_scaler.fit_transform(test_data)
        #     Dense(32, input_dim=784),
        #     Activation('relu'),
        #     Dense(10),
        #     Activation('softmax'),
        # ])
        # # Another way to define your optimizer
        # rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        # # We add metrics to get more results you want to see
        # model.compile(optimizer=rmsprop,
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])
        #
        # print('Training ------------')
        # # Another way to train the model
        # model.fit(X_train, y_train, batch_size=32)
        # # We add metrics to get more results you want to see
        # model.compile(optimizer=rmsprop,
        #               loss='categorical_crossentropy',
        #               metrics=['accuracy'])



if __name__ == "__main__":
    # for i in range(0,11):
    # appName = 'SLife'
    # load_fn = 'Image/mfiles/result/' + appName + '.mat'
    # appID = "10010000000032"
    # classifier(load_fn, appName, appID)
    rb = open_workbook('/Users/liyuying/Desktop/result.xls')
    rs = rb.sheet_by_index(0)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    row = 2
    col = 1
    appName = ['MyListening', 'HuJiang', 'HuaWei', '10010000000024', 'Game2048', 'Slife']
    appID = ['10010000000019', '10010000000037', '10010000000036', '10010000000024', '10010000000017', '10010000000032']

    # appName = ['MyListening']
    # appID = ['10010000000019']

    for num in range(0, 6):
        print appName
        classifier(appName, appID, row, col)
        num += 1
        row +=1

    wb.save('/Users/liyuying/Desktop/result.xls')
