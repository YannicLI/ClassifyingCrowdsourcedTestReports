#encoding=utf-8
'''
Created on Mar 18, 2018

@author: liyuying
'''
import wordSegmentation

def CreateDictionary(appID):
    dict = {}
    sortedDict = {}
    k = 0
    for key,value in wordSegmentation.WordSegmentation(appID).items():
        # print key
        for i in value:
            #print i
            if i in dict.keys():
                dict[i] = dict[i] + 1
            else:
                dict[i] = 1
    
    tmp = sorted(dict.iteritems(),key=lambda x:x[1],reverse=True)
    for j in tmp:
        # print j[0]
        # print j[1]
        sortedDict[j[0]] = j[1]
        k = k + 1
        if k ==100:
            break
    #
    # for q in sortedDict:
    #     print q
    #     print sortedDict[q]

    return sortedDict

if __name__ == "__main__":
    appID = '10010000000019'
    CreateDictionary(appID)


