import json
from gensim.models import KeyedVectors

def evaluate(engine):
    print("Beginning evaluation on CLEF dataset")
    queries=json.load(open("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\CLEF\\questionVectors.json","r"))
    precision=0
    recall=0
    totalQueries=0#To track total number of queries between the different languages
#-----------------------------Calculate the score based on the queries------------------------------------------------------------------------------------
    for i in queries.keys():#For each query
        relevantDocuments=queries[i]["relevant documents"]#Find all relevant documents
        if len(relevantDocuments)==0:
            continue#Ignore query if no relevant documents
        if "en text" in queries[i].keys():#If the query is in english
            documentsRetrievedEn=engine.search(queries[i]["en text"],lang="en")#Search the query with the engine
            if len(documentsRetrievedEn)==0:
                continue
            relevantRetrieved=len(set(documentsRetrievedEn).intersection(relevantDocuments))#Find the number of relevant documents retrieved
            precision+=relevantRetrieved/len(documentsRetrievedEn)#Add to precision
            recall+=relevantRetrieved/len(relevantDocuments)#Add to recall
            totalQueries+=1
        if int(i)%10==0:
            print("{:.2f}% complete".format(int(i)*100/(len(list(queries.keys()))*2)))#Print the percentage complete
#Split into 2 loops to save on computation cost of switching language models
    for i in queries.keys():
        relevantDocuments=queries[i]["relevant documents"]
        if len(relevantDocuments)==0:
            continue
        if "es text" in queries[i].keys():
            documentsRetrievedEs=engine.search(queries[i]["es text"],lang="es")
            if len(documentsRetrievedEs)==0:
                continue
            relevantRetrieved=len(set(documentsRetrievedEs).intersection(relevantDocuments))
            precision+=relevantRetrieved/len(documentsRetrievedEs)
            recall+=relevantRetrieved/len(relevantDocuments)
            totalQueries+=1
        if (int(i)+len(queries.keys()))%10==0:
            print("{:.2f}% complete".format((int(i)+len(queries.keys()))*100/(len(list(queries.keys()))*2)))
    print("100% complete")
    return (precision/totalQueries,recall/totalQueries)

#-----------------------Convert FastText .vec files to gensim KeyedVectors---------------------------------------------------------------------------
def vec2File(f,lang,path):
    #f is the vector file
    #lang is the language of the model
    #path is the directory to save the file to
    print("Extracting Data")
    model=KeyedVectors.load_word2vec_format(f)
    print("Saving")
    model.save(path+"aligned{}Vecs".format(lang))


