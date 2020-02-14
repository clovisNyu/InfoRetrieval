import os
import pickle
import numpy as np
from corpus import token
from corpus import corpus
from gensim.models import KeyedVectors
import nltk
from sys import maxsize
import time
import sys

sys.path.insert(1, 'C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\CLEF')

from utils import evaluate


#---------------------------Engine class that holds all methods required for search----------------------------------------------------
class engine():
    def __init__(self):
        self.languageModels={}#Stores the paths to the prealigned vectors for specific languages
        self.model=None#Current language model
        self.currentLanguage=None#Current language
        self.corpusDirectories=[]#Stores the directories to the specific corpus being used
        self.corpus=None#Current corpus

#To load a model, it must be saved via gensim.models.KeyedVectors, you can do this via the vec2File function in utils.py
    def loadModel(self,directory=None,lang=None):#Method that loads a language model
        if directory!=None:#If the user specifies a directory
            if isinstance(directory,str):#If the directory specified is a string
                if os.path.isfile(directory):#and it correctly leads to a file
                    if directory not in self.languageModels.values():#and the directory is not already in the database of languages
                        self.languageModels[lang]=directory#add language to the database of languages
                    self.model=KeyedVectors.load(directory,mmap="r")#load the model with memory maps to save computation and memory
                    self.currentLanguage=lang#set the current language accordingly
                    print("Model Loaded")
                else:#In all invalid cases, simply do not load
                    print("Directory does not exist")
            else:
                print("Invalid directory")
        else:#if not directory is specified, simply load the last used model
            if len(list(self.languageModels.keys()))!=0:
                directory=self.languageModels[list(self.languageModels.keys())[-1]]
                self.currentLanguage=list(self.languageModels.keys())[-1]
                print("Latest model Loaded")
            else:
                print("No path specified")

    def loadCorpus(self,directory=None):#Method to load a corpus
        if directory!=None:#If a directory is specified
            if isinstance(directory,str):#if directory is a string
                if os.path.isfile(directory):#and is a valid file path
                    if directory not in self.corpusDirectories:#if the path is not already in the database
                        self.corpusDirectories.append(directory)#add it to the database
                    self.corpus=pickle.load(open(directory,"rb"))#Load the corpus
                    print("Corpus Loaded")
                else:#In all invalid cases, simply do not load
                    print("Directory does not exist, corpus not loaded")
            else:
                print("Invalid directory, corpus not loaded")
        else:#If no directory is specified
            if len(self.corpusDirectories)!=0:#If there are available corpora
                self.corpus=pickle.load(open(self.corpusDirectories[-1],"rb"))#load the latest corpus
                print("Latest corpus loaded")
            else:
                print("No available corpus")

    def search(self,query,depth=10,lang="en",feedback=0):#Search method
        #depth changes the number of results returned
        #language indicates which model to use to search
        #feedback changes the number of psuedo-relevance feedback loops to use
        stopWords=[line.split()[0] for line in open("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\SMART Stopwords.txt","r")]
        start=time.time()
        if lang not in self.languageModels.keys():#If the language is not found
            print("Language model not found")
            return []#No search can be performed, return without result
        if lang!=self.currentLanguage:#If the language required for search is not current language
            self.model=KeyedVectors.load(self.languageModels[lang],mmap="r")#Load the appropriate language
            self.currentLanguage=lang
        #Form the query vector
        qVec=np.array(np.zeros((1,300)))[0]
        invalidCount=0
        queryTerms=[]
        for term in query.split(" "):
            cleanTerm="".join(a.lower() for a in term if a.isalnum() or a=="-" or a=="_")
            if cleanTerm not in stopWords:
                queryTerms.append(cleanTerm)
        for qTerm in queryTerms:
            try:#If not OOV
                qVec+=self.model[qTerm]*query.count(qTerm)*np.log10(len(self.corpus.documents)/self.corpus.tokens[qTerm].df)#Compute tf-idf weights
            except KeyError:#OOV
                invalidCount+=1
                continue
        if invalidCount==len(query.split()):#If the whole query is OOV
            print("Query OOV")
            return []#No meaningful search can be performed
        qVec=qVec/(len(query.split())-invalidCount)#Average the term vectors
        rankedDocs=[]
        index=0
        for doc in self.corpus.documentVectors:#For each document
            if np.linalg.norm(np.array(doc[1]))!=0 and np.linalg.norm(qVec)!=0:
                score=np.dot(np.array(doc[1]),qVec)/np.linalg.norm(np.array(doc[1]))/np.linalg.norm(qVec)#Find the cosine similarity
            else:
                score=0
            rankedDocs.append((doc[0],score,index))
            index+=1
        rankedDocs.sort(key=lambda x:x[1], reverse=True)#Sort by highest score first
        if feedback==0:#If psuedo-relevance feedback has been performed sufficiently
            
            # topDocs=[(doc[0],self.corpus.documents[doc[2]]) for doc in rankedDocs[:depth*2]]
            # topDocs=list(set(topDocs))
            # reranked=self.WMDRerank(topDocs,queryTerms)
            # if __name__=="__main__":
            #     print("{} results found in {:.2f}s".format(depth,time.time()-start))
            # return [doc[0] for doc in reranked[:depth]]#Return the documents
            return [doc[0] for doc in rankedDocs[:depth]]

        else:#Otherwise, perform psuedo-relevance feedback
            feedback-=1
            topDocs=[]
            for doc in rankedDocs[:depth]:
                topDocs.append((doc[0],self.corpus.documents[doc[2]]))
            additionalQueryTerms=self.expandQuery(topDocs,queryTerms)#Find expanded query
            for term in additionalQueryTerms:
                query+=" "+term+" "
            return self.search(query,depth=depth,lang=lang,feedback=feedback)#Reperform the search

    def expandQuery(self,documents,queryTerms):#Psuedo-relevance feedback for query expansion
        print("Entering query expansion")
        # documents=self.WMDRerank(documents,queryTerms)
        words=[]#All words in the documents
        stopWords=[line.split()[0] for line in open("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\SMART Stopwords.txt","r")]#File containing stopwords
        for url,document in documents:
            for word in document.split():
                if word not in stopWords:
                    words.append(word)
        scores=[]
        uniqueWords=list(set(words))#All unique words
        print("Expanding Query")
        for word in uniqueWords:
            scores.append((word,words.count(word)*self.corpus.tokens[word].df))#Calculate tf-idf scores for each word
        scores.sort(key=lambda x:x[1],reverse=True)
        return [word[0] for word in scores[:1]]#Return only the terms with the highest tf-idf scores
    def WMDRerank(self,documents,queryTerms):
        print("Reranking")
        stopWords=[line.split()[0] for line in open("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\SMART Stopwords.txt","r")]
        documentDistances=[]
        for url,document in documents:
            documentDistance=maxsize
            sentences=self.corpus.documentSentences[url]
            for sentence in sentences:
                averageSentenceDistance=0
                for term in queryTerms:
                    queryTermDistance=maxsize
                    for sentenceTerm in sentence.split(" "):
                        cleanSentenceTerm="".join(a.lower() for a in sentenceTerm if a.isalnum() or a=="-" or a=="_")
                        if cleanSentenceTerm not in stopWords:
                            distance=self.model.wmdistance(cleanSentenceTerm,term)
                            if distance<queryTermDistance:
                                queryTermDistance=distance
                    if queryTermDistance!=maxsize:
                        averageSentenceDistance+=queryTermDistance
                if averageSentenceDistance!=0 and averageSentenceDistance/len(queryTerms)<documentDistance:
                    documentDistance=averageSentenceDistance/len(queryTerms)
            documentDistances.append((url,document,documentDistance))
        documentDistances.sort(key=lambda x:x[2])
        return [(doc[0],doc[1]) for doc in documentDistances]

            
    def export(self,fname,fpath=""):
        print("Exporting")
        if fpath!="":
            if not os.path.isdir(fpath):
                os.makedirs(fpath)
            if fpath[-1]!="/" and fpath[-1]!="\\":
                fpath+="\\"
        fullPath=fpath+fname+".pkl"
        pickle.dump(self,open(fullPath,"wb"))
        print("Successfully exported")
modelPath="C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Program Files\\FastText\\Models\\alignedEnVecs"
modelLang="en"
corpusPath="C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\CLEF\\FastText\\CLEF-FastText.pkl"
modelPathEs="C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Program Files\\FastText\\Models\\alignedEsVecs"

testEngine=engine()

testEngine.loadCorpus(corpusPath) #Insert path to corpus object
testEngine.loadModel(modelPath,lang=modelLang) #Insert path to model
testEngine.loadModel(modelPathEs,lang="es")


# precision,recall=singleEval(testEngine,210)
# print("Precision:{}\nRecall:{}".format(precision,recall))

precision,recall=evaluate(testEngine)
print("Average Precision:{}\nAverage Recall:{}".format(precision,recall))
