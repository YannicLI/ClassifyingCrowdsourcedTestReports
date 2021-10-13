#encoding=utf-8
'''
Created on Mar 18, 2018

@author: liyuying
'''
from __future__ import division
import dict
import math
import copy
import MySQLdb
import numpy
import numpy as np
import getKeyWords

def getVector(appID):
    conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='root',port=3306)

    cur=conn.cursor()
    conn.select_db('kikbug')

    cur.execute("SET character_set_client = UTF8MB4;")
    cur.execute("SET character_set_connection = UTF8MB4;")
    cur.execute("SET character_set_results = UTF8MB4;")
    cur.execute("SET character_set_server = UTF8MB4;")


    dictionary = {}
    allVectors = {}


    vector = []
    i = 0
    while i < 100:
        vector.append(0.0)
        i = i + 1
    keywords = {}
    sum_oneWordInOneReport = {}#某一bug报告中一个词的数量
    sum_allWordsInOneReport = {}#所有报告中一个词的数量
    sum_allReports = 0#bug报告数量
    sum_oneWordInAllReports = {}#出现过一个词的bug报告数量
    reportVector = {}
    flag = 0
    tf_idfResult = {}
    dictionryPosition = {}
    tf = 0.0
    idf = 0.0

    for keyword,sum in dict.CreateDictionary(appID).items():
        dictionary[keyword] = sum
        dictionryPosition[keyword] = flag
        flag = flag + 1
    for key,value in getKeyWords.getKeywords(appID).items():
        sum_allReports = sum_allReports + 1 #bug报告数量
        keywords[key] = value


    for key,value in keywords.items():
        sum_oneWordInOneReport = {}#某一bug报告中一个词的数量
        sum_allWordsInOneReport = 0#一个bug报告中所有词的数量
        sum_oneWordInAllReportsN = {}#出现过一个词的bug报告数量，算了n次
        sum_oneWordInAllReports = {}#出现过一个词的bug报告数量
        # print key
        sum_allWordsInOneReport = len(value)
        for oneWord in value:
            if oneWord in sum_oneWordInAllReportsN.keys():
                sum_oneWordInOneReport[oneWord] = sum_oneWordInOneReport[oneWord] + 1
            else :
                sum_oneWordInOneReport[oneWord] = 1
            for key2,value2 in keywords.items():
                    if oneWord in value2:
                        if oneWord in sum_oneWordInAllReportsN.keys():
                            sum_oneWordInAllReportsN[oneWord] = sum_oneWordInAllReportsN[oneWord] + 1
                        else :
                            sum_oneWordInAllReportsN[oneWord] = 1
        '''                    
        print "某一bug报告中一个词的数量"
        for a in sum_oneWordInOneReport:
            print a
            print sum_oneWordInOneReport[a]
        
        print "一个bug报告中所有词的数量"
        print sum_allWordsInOneReport
        
    
        print "出现过一个词的bug报告数量"
        '''
        for c in sum_oneWordInAllReportsN:
            #print c
            sum_oneWordInAllReports[c] = sum_oneWordInAllReportsN[c]/sum_oneWordInOneReport[c]
            #print sum_oneWordInAllReports[c]

        for oneWordInVector in sum_oneWordInOneReport:
            tf = sum_oneWordInOneReport[oneWordInVector]/sum_allWordsInOneReport
            #print "tf:",tf
            idf = math.log10(sum_allReports/sum_oneWordInAllReports[oneWordInVector])
            #print "idf:",idf
            tf_idf = tf * idf
            #print dictionryPosition[oneWordInVector]
            #print "oneWordInVector"+oneWordInVector
            vector[dictionryPosition[oneWordInVector]] = tf_idf
        #print "The vector is"
        #print vector
        tf_idfResult[key] = copy.deepcopy(vector)
        i = 0
        while i < 100:
            vector[i] = 0.0
            i = i + 1
    '''        
    print "bug报告数量"
    print sum_allReports
    '''

    oneDis = []
    allDis = {}
    t = 0
    for y in tf_idfResult:
        y_vec = np.array(tf_idfResult[y])
        if tf_idfResult[y].count(0.0) < 100 :
            for x in tf_idfResult:
                if tf_idfResult[y].count(0.0) == 100 or tf_idfResult[x].count(0.0) == 100:
                    oneDis.append(99)
                else:
                    x_vec = np.array(tf_idfResult[x])
                    dist = numpy.sqrt(numpy.sum(numpy.square(y_vec - x_vec)))
                    oneDis.append(dist)
                #print dist
                dist = 0.0
        else:
            t = 0
            while t < 100:
                oneDis.append(99)
                t = t+1
        allDis[y] = oneDis
        oneDis = []

        try:
            sql = "update set vector = %s "
            sql = "insert into vector (bug_id,text_vector) values(%s,%s)"
            cur.execute(sql,[str(y),str(tf_idfResult[y])])
            conn.commit()
        except:
            print "Wrong:" + str(y)


if __name__ == "__main__":
    appID = "10010000000017"
    getVector(appID)