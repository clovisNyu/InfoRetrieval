import json
import requests
import bs4
import numpy as np
import time
from gensim.models import KeyedVectors

#This program extracts queries from the data provided by CLEF. It currently extracts English and Spanish queries but can easily be editted
#to extract other languages

#Load the appropriate models. It should be saved via KeyedVectors methods. Use the vec2File function in utils.py
englishPath="alignedEnVecs"
spanishPath="alignedEsVecs"
modelEn = KeyedVectors.load(englishPath,mmap="r")
modelEs = KeyedVectors.load(spanishPath,mmap="r")
#Load file containing stopwords
stopWords=[line.split()[0] for line in open("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\SMART Stopwords.txt","r")]

def vectorize(string,model):#Function to vectorize string using the given model
    clean=''.join(a.lower() for a in string if a not in ")({,.[]/?!@#$%^&*`~+*\"\':;><}")
    terms = clean.split()
    vector = np.zeros((1,300))
    for term in terms:
        if term not in stopWords:
            try:
                vector+=model[term]
            except KeyError:
                continue
    if len(terms)!=0:
        return vector/len(terms)
    else:
        return vector


if __name__=="__main__":
    start=time.time()

    print("Retrieving Information")
    rawCorpus=requests.get("https://raw.githubusercontent.com/ag-sc/QALD/master/9/data/qald-9-train-multilingual.json").content

    print("Loading Information")
    corpusDictionary=json.loads(rawCorpus)
    questionList=corpusDictionary["questions"]

    questionVectors={}
    print("Vectorizing")
    for question in questionList:
        questionVectors[question["id"]]={}
        for content in question["question"]:
            if content["language"]=="en" or content["language"]=="es":
                try:
                    if content["language"]=="en":
                        questionVectors[question["id"]][content["language"]]=vectorize(content["string"],modelEn).tolist()[0]
                        questionVectors[question["id"]]["en text"]="".join(a.lower() for a in content["string"] if a not in ")({,.[]/?!@#$%^&*`~+*\"\':;><}")
                    elif content["language"]=="es":
                        questionVectors[question["id"]][content["language"]]=vectorize(content["string"],modelEs).tolist()[0]
                        questionVectors[question["id"]]["es text"]="".join(a.lower() for a in content["string"] if a not in ")({,.[]/?!@#$%^&*`~+*\"\':;><}")

                    questionVectors[question["id"]]["relevant documents"]=[]
                except KeyError:#Some queries don't have relevant documents as answers, in which case we ignore the query
                    continue
                for ans in question["answers"]:
                    try:
                        for entry in ans["results"]["bindings"]:
                            try:
                                questionVectors[question["id"]]["relevant documents"].append(entry["uri"]["value"])
                            except KeyError:#Some queries don't have relevant documents as answers, in which case we ignore the query
                                continue
                    except KeyError:#Some queries don't have relevant documents as answers, in which case we ignore the query
                        continue
    print("Saving Information")
    #Persistance to disk in the form of a JSON
    with open("questionVectors.json","w") as fp:
        json.dump(questionVectors,fp,indent=4)

    print("Time Elapsed: {:.2f}".format(time.time()-start))
