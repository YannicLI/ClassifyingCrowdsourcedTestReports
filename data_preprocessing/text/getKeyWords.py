#encoding=utf-8
'''
Created on Mar 18, 2018

@author: liyuying
'''
import wordSegmentation
import dict

def getKeywords(appID):
    dictionary = {}
    words = {}
    
    for keyword,sum in dict.CreateDictionary(appID).items():
        dictionary[keyword] = sum
    
    for key,value in wordSegmentation.WordSegmentation(appID).items():
        tmp = []
        for oneWord in value:
            for keyword,sum in dictionary.items():
                if oneWord == keyword:
                    # print "key:"
                    # print key
                    tmp.append(oneWord)
                    # print "oneWord:"
                    # print oneWord
                    break
        words[key] = tmp
    # for q in words:
    #     print q
    #     for w in words[q]:
    #         print w
    return words


if __name__ == "__main__":
    appID = "10010000000019"
    getKeywords(appID)
