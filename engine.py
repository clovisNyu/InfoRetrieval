import os
import pickle
import numpy as np
from corpus import token
from corpus import corpus
from gensim.models import KeyedVectors
# from utils import evaluate


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

    def search(self,query,depth=10,lang="en",feedback=1):#Search method
        #depth changes the number of results returned
        #language indicates which model to use to search
        #feedback changes the number of psuedo-relevance feedback loops to use

        if lang not in self.languageModels.keys():#If the language is not found
            print("Language model not found")
            return#No search can be performed, return without result
        if lang!=self.currentLanguage:#If the language required for search is not current language
            self.model=KeyedVectors.load(self.languageModels[lang],mmap="r")#Load the appropriate language
            self.currentLanguage=lang
        #Form the query vector
        qVec=np.array(np.zeros((1,300)))[0]
        invalidCount=0
        for qTerm in query.split():
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
        rankedDocs.sort(key=lambda x:x[1], reverse=True)#Sort by highest score first
        if feedback==0:#If psuedo-relevance feedback has been performed sufficiently
            return [doc[0] for doc in rankedDocs[:depth]]#Return the documents
        else:#Otherwise, perform psuedo-relevance feedback
            feedback-=1
            topDocs=[]
            for doc in rankedDocs[:depth]:
                topDocs.append(self.corpus.documents[doc[2]])
            additionalQueryTerms=self.expandQuery(topDocs)#Find expanded query
            for term in additionalQueryTerms:
                query+=" "+term+" "
            return self.search(query,depth=depth,lang=lang,feedback=feedback)#Reperform the search

    def expandQuery(self,documents):#Psuedo-relevance feedback for query expansion
        words=[]#All words in the documents
        stopWords=[line.split()[0] for line in open("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\SMART Stopwords.txt","r")]#File containing stopwords
        for document in documents:
            for word in document.split():
                if word not in stopWords:
                    words.append(word)
        scores=[]
        uniqueWords=list(set(words))#All unique words
        for word in uniqueWords:
            scores.append((word,words.count(word)*self.corpus.tokens[word].df))#Calculate tf-idf scores for each word
        scores.sort(key=lambda x:x[1],reverse=True)
        return [word[0] for word in scores[:1]]#Return only the terms with the highest tf-idf scores
        
modelPath="C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Program Files\\FastText\\Models\\alignedEnVecs"
modelLang="en"
corpusPath="C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\CLEF\\FastText\\CLEF-FastText.pkl"


testEngine=engine()

testEngine.loadCorpus(corpusPath) #Insert path to corpus object
testEngine.loadModel(modelPath,lang=modelLang) #Insert path to model
print(testEngine.search("board games by gmt",depth=10,feedback=1)) #Example search

# precision,recall=evaluate(testEngine)
# print("Average Precision:{}\nAverage Recall:{}".format(precision,recall))
