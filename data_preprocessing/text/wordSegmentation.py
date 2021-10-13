#encoding=utf-8
'''
Created on Mar 18, 2018

@author: liyuying
'''
import MySQLdb
import json
import time
import jieba.posseg as pseg


def WordSegmentation(appID):
    
    # url_get_base = "http://api.ltp-cloud.com/analysis/?"
    # api_key = 'e7F5z5s2wRajoc2apY8hOSLFUf4Uetcg3uZbqAKx'
    # format = 'json'
    # pattern = 'pos'
    filter = ['v','n', 'vn']
    conn=MySQLdb.connect(host='127.0.0.1',user='root',passwd='root',port=3306)
    
    cur=conn.cursor()
    conn.select_db('kikbug')
    
    cur.execute("SET character_set_client = UTF8MB4;")
    cur.execute("SET character_set_connection = UTF8MB4;")
    cur.execute("SET character_set_results = UTF8MB4;")    
    cur.execute("SET character_set_server = UTF8MB4;")
    sql = "select b.id,b.description from bug as b inner join report r on b.report_id=r.id where r.case_id=%s order by b.id"
    cur.execute(sql,[appID])
    res = cur.fetchall()
    conn.commit()
    words= {}
    # f = open('E://Project/classification/ClassifyingCrowdsourcedTestReports/result/'+appID+'word.txt','a+')
    for r in res:
        text = ''
        text = r[1].replace('\n','')
        # print str(r[0])
        if text != '':
            key = str(r[0])
            segmentation = pseg.cut(text)
            tmp = []
            for w in segmentation:
                # print w.word, w.flag
                if w.flag in filter:
                    # print w.word, w.flag
                    oneWord = w.word
                    tmp.append(oneWord+" ")
            words[key] = tmp

    # for q in words:
    #     print q
    #     for w in words[q]:
    #         print w
    return words

# if __name__ == "__main__":
#     appID = "10010000000019"
#     WordSegmentation(appID)