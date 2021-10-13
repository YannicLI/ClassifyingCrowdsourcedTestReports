#encoding=utf-8
'''
Created on Mar 21, 2018

@author: liyuying
'''

import scipy.io as sio
import os
import MySQLdb
from sklearn.naive_bayes import GaussianNB

def categoryStr(num):
    if num == 1:
        return "安全"
    if num == 2:
        return "安装失败"
    if num == 3:
        return "不正常退出"
    if num == 4:
        return "登录异常"
    if num == 5:
        return "更新异常"
    if num == 6:
        return "功能不完整"
    if num == 7:
        return "性能"
    if num == 8:
        return "页面布局缺陷"
    if num == 9:
        return "用户体验"
    if num == 10:
        return "注册异常"
    if num == 11:
        return "其他"

def categoryNum(name):
    if name == "安全":
        return 1
    if name == "安装失败":
        return 2
    if name == "不正常退出":
        return 3
    if name == "登录异常":
        return 4
    if name == "更新异常":
        return 5
    if name == "功能不完整":
        return 6
    if name == "性能":
        return 7
    if name == "页面布局缺陷":
        return 8
    if name == "用户体验":
        return 9
    if name == "注册异常":
        return 10
    if name == "其他":
        return 11

def getVector(load_fn,appName,appID):

    load_data = sio.loadmat(load_fn)
    lst = ['0.0' for n in range(4200)]

    vector = load_data.values()[0]
    conn = MySQLdb.connect(host='127.0.0.1', user='root', passwd='root', port=3306)
    cur = conn.cursor()
    conn.select_db('kikbug')
    cur.execute("SET character_set_client = UTF8MB4;")
    cur.execute("SET character_set_connection = UTF8MB4;")
    cur.execute("SET character_set_results = UTF8MB4;")
    cur.execute("SET character_set_server = UTF8MB4;")

    path = './mfiles/image/'+appName
    # vec_path = './mfiles/result/' + appName + '.txt'
    # vec_fr = open(vec_path,'a+')
    train_data = []
    test_data = []
    sql = "select bug_id from training_set where case_id = %s"
    cur.execute(sql,[appID])
    res = cur.fetchall()
    conn.commit()
    training_id = []
    test111 = 0
    for i in res:
        one_id = str(i).replace('L', '').replace('(', '').replace(')', '').replace(',', '')
        training_id.append(one_id)

    train = {}
    test = {}

    for root in os.walk(path):
        file_list = str(root).replace('[','').replace(']','').replace('\'','').split(',')
        file = sorted(file_list)
        del file[0]
        # del file[len(file)-1]
        flag = 0
        for one in file:  # 当前目录路径
            # print str(one)
            # test111 = test111 + 1           
            if flag == 0:
                bug_id_p = one.split('-')[0].replace(" ",'')
                tmp_vector = vector[flag]
                flag = flag + 1
                # print str(bug_id_p)
                # print tmp_vector
                sql = "update vector set image_vector = %s where bug_id = %s"
                
                cur.execute(sql, [str(tmp_vector.tolist()),bug_id_p])
                conn.commit()
                continue
            bug_id = one.split('-')[0].replace(" ",'')
            if bug_id == bug_id_p:
                tmp_vector = vector[flag] + tmp_vector
                flag = flag + 1
                continue
            else:
                # fr = open('./mfiles/result/'+appName+'.txt',"a+")
                # print bug_id_p
                # print str(tmp_vector.tolist())
                print str(bug_id_p)
                sql = "update vector set image_vector = %s where bug_id = %s"
                
                cur.execute(sql, [str(tmp_vector.tolist()),bug_id_p])
                conn.commit()
                if bug_id_p.replace(' ','') in training_id:
                    tmp_vector.tolist()
                    train[bug_id_p] = [float(x) for x in tmp_vector.tolist()]
                    # train_data.append(tmp_vector.tolist())
                else:

                    test_data.append([float(x) for x in tmp_vector.tolist()])

                    #test[bug_id_p] = [float(x) for x in tmp_vector.tolist()]
                # print flag
                # print vector[flag]
                # tmp_vector = vector[flag]
                print flag
                bug_id_p = bug_id
            flag = flag + 1
    # vec_fr.close()

    # # print "test111"
    # # print test111
    # sql = "select bug_id,bug_category from training_set where case_id = %s ORDER BY bug_id"
    # cur.execute(sql, [appID])
    # res = cur.fetchall()
    # conn.commit()
    # train_target = []
    # for one in res:
    #     #print ' ' + str(one[0]).replace('L', '').replace('(', '').replace(')', '').replace(',', '')
    #     #print train.keys()
    #     if str(one[0]).replace('L', '').replace('(', '').replace(')', '').replace(',', '') in train.keys():
    #         train_data.append(train[str(one[0]).replace('L', '').replace('(', '').replace(')', '').replace(',', '')])
    #     else:
    #         train_data.append([float(x) for x in lst])
    #     train_target.append(categoryNum(one[1]))

    # #NB
    # print len(train_data)
    # # print len(train_target)
    # print len(test_data)

    # gnb = GaussianNB().fit(train_data, train_target)
    # result = gnb.predict(test_data)

    # return result
    return

def insertResult(result,appID):
    conn = MySQLdb.connect(host='127.0.0.1', user='root', passwd='root', port=3306)

    cur = conn.cursor()
    conn.select_db('kikbug')

    cur.execute("SET character_set_client = UTF8MB4;")
    cur.execute("SET character_set_connection = UTF8MB4;")
    cur.execute("SET character_set_results = UTF8MB4;")
    cur.execute("SET character_set_server = UTF8MB4;")

    sql = "select bug_id, img_url from testing_set where case_id = %s ORDER BY bug_id"
    cur.execute(sql, [appID])
    res = cur.fetchall()
    conn.commit()
    test_null = []
    for bug in res:
        if str(bug[1]) == "":
            test_null.append(str(bug[0]).replace('L', '').replace('(', '').replace(')', '').replace(',', ''))
    # for a in test_null:
    #    print a

    sql = "select bug_id from testing_set where case_id = %s ORDER BY bug_id"
    cur.execute(sql, [appID])
    res = cur.fetchall()
    conn.commit()
    flag = 0

    for one in res:
        sql = "update testing_set set image_category = %s where bug_id = %s"
        bug_id = str(one).replace('L', '').replace('(', '').replace(')', '').replace(',', '')
        #print bug_id
        if bug_id in test_null:
            continue
        test_category = categoryStr(result[flag])
        # print "bug_id"
        # print bug_id
        cur.execute(sql,[test_category,bug_id])
        conn.commit()
        flag = flag + 1

if __name__ == "__main__":
    appName = "Game2048"
    load_fn = 'E://Project/classification/ClassifyingCrowdsourcedTestReports/ClassifyingCrowdsourcedTestReports/Image/mfiles/result/'+appName+'.mat'
    appID = "10010000000017"
    result = getVector(load_fn,appName,appID)
    # insertResult(result,appID)
