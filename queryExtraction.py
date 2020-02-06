import json
import requests
import bs4
import numpy as np
import time
from gensim.models import KeyedVectors

#Load the appropriate models. It should be saved via KeyedVectors methods. Use the vec2File function in utils.py
modelEn = KeyedVectors.load("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Program Files\\FastText\\Models\\alignedEnVecs",mmap="r")
modelEs = KeyedVectors.load("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Program Files\\FastText\\Models\\alignedEsVecs",mmap="r")
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
                    elif content["language"]=="es":
                        questionVectors[question["id"]][content["language"]]=vectorize(content["string"],modelEs).tolist()[0]
                    questionVectors[question["id"]]["text"]="".join(a.lower() for a in content["string"] if a not in ")({,.[]/?!@#$%^&*`~+*\"\':;><}")
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