import json
import requests
import bs4
import numpy as np
import time
from gensim.models import KeyedVectors
import multiprocessing


#Program extracts information from CLEF's JSON at https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-train-multilingual.json
#Do note that this program takes a while to complete.

modelPath="/alignedEnVecs"#Path to file produced by utils.vec2File()
model = KeyedVectors.load(modelPath,mmap="r")#Input file path to appropriate model
stopWords=[line.split()[0] for line in open("SMART Stopwords.txt","r")]

#Various statistics that may be useful
df={}#Document term frequency
ctf={}#Collection term frequency
mtf={}#Mean term frequency (Like collection term frequency but normalized for document length)
totalDocCount=0#Total number of documents


def vectorize(string,model):#Function calculates the tf-idf averaged vectors of whatever string is passed using whatever mdoel is passed
    terms = string.lower().split()
    vector = np.zeros((1,300))
    totalTfIdf=0
    for term in terms:
        if term not in stopWords:
            try:
                tfidf=string.count(term)*np.log10(totalDocCount/df[term])
                vector += model[term]*tfidf
                totalTfIdf+=tfidf
            except KeyError:
                continue
    if totalTfIdf!=0:
        return vector/totalTfIdf
    else:
        return vector



def vectorizeCorpus(args):# Combined all arguments into 1 "args" argument to make working with Multiprocessing easier later on
    #Function scrapes information from the website
    URL=args[0]
    documentID=args[1]
    testContent=requests.get(URL).content
    soup=bs4.BeautifulSoup(testContent,"lxml")
    try:
        information = soup.find("p").text
        return (documentID,URL,information)
    except AttributeError:
        return (documentID,URL,"")


if __name__=="__main__":
    start=time.time()
    print("Retrieving Information")
    rawCorpus=requests.get("https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-train-multilingual.json").content#Raw information from the CLEF JSON
    print("Loading Information")
    corpusDictionary=json.loads(rawCorpus)
    questionList=corpusDictionary["questions"]
    print("Building Corpus")
    corpus=[]
    idCounter=0
    for question in questionList:
        try:
            for ans in question["answers"]:
                for url in ans["results"]["bindings"]:
                    idCounter+=1
                    corpus.append((url["uri"]["value"],idCounter))
        except KeyError:#Some queries don't have relevant documents as answers, in which case we ignore the query
            continue
    # corpusVectors={}
    print("Vectorizing")
    p=multiprocessing.Pool()#Use multiprocessing to speed up the scraping of information
    docs=p.map(vectorizeCorpus,corpus)
    p.close()
    p.join()#Pause program until the subprocesses are done
    docsClean=[]#Fasttext uses only lower case and avoids special characters
    #The following loop formats the content accordingly and calculates the document frequency
    for documentID,URL,doc in docs:
        docClean=""
        for term in doc.split():
            termClean="".join(a.lower() for a in term if a not in ")({,.[]/?!@#$%^&*`~+*\"\':;><}")
            docClean+=termClean+" "
            if termClean in df.keys():
                df[termClean]+=1
            else:
                df[termClean]=1
        docsClean.append((documentID,URL,docClean))
    totalDocCount=len(docsClean)
    documentData=[]
    #The following loop stores document data as a list of dictionaries in preparation for disk persistance later on
    for documentID,URL,doc in docsClean:
        documentData.append({"id":documentID,"url":URL,"text":doc,"vector":vectorize(doc,model).tolist()[0]})

    #The following loop calculates the collection and mean term frequency. 
    for doc in documentData:
        for term in doc["text"].split():
            if term in ctf.keys():
                ctf[term]+=doc["text"].count(term)
            else:
                ctf[term]=doc["text"].count(term)
            if term in mtf.keys():
                mtf[term]+=doc["text"].count(term)/len(doc["text"].split())
            else:
                mtf[term]=doc["text"].count(term)/len(doc["text"].split())
    for term in mtf.keys():
        mtf[term]=mtf[term]/df[term]
    
    #Final dictionary to be saved
    corpusInfo={"documents":documentData,"statistics":{"document term frequency":df,"collection term frequency":ctf,"mean term frequency":mtf,"total document count":totalDocCount}}


    print("Saving Information")
    #Disk persistance in the form of JSON
    with open("corpusVectors.json","w") as fp:
        json.dump(corpusInfo,fp,indent=4)
    print("Time Elapsed: {:.2f}".format(time.time()-start))

        


