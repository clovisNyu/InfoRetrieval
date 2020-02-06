import pickle
import os
import json

class token():#Token class to store information about each token
    def __init__(self,text,df,mtf,ctf):
        self.text=text
        self.df=df
        self.mtf=mtf
        self.ctf=ctf

class corpus():#Corpus class to store the documents
    def __init__(self,corpusJSON):
        self.corpusJSON=corpusJSON#Corpus object is formed via a corpus json which stores the information
        documents=json.load(open(self.corpusJSON,"r"))
        self.documents=[]
        self.documentVectors=[]
        for doc in documents["documents"]:#Form the list of documents and vectors
            self.documents.append(doc["text"])
            self.documentVectors.append((doc["url"],doc["vector"]))
        self.vocab=list(documents["statistics"]["document term frequency"].keys())#Form the vocabulary
        self.tokens={}#Stores token information
        #Extract token information from json
        docFrequency=documents["statistics"]["document term frequency"]
        cTermFrequency=documents["statistics"]["collection term frequency"]
        mTermFrequency=documents["statistics"]["mean term frequency"]
        #Store token information in self.tokens
        for term in self.vocab:
            try:#If term is found in the json information
                self.tokens[term]=(token(term,docFrequency[term],mTermFrequency[term],cTermFrequency[term]))#Create a token
            except KeyError:
                continue
        self.vocab=list(self.tokens.keys())#To remove terms not found in the token information in the json

    def export(self,directory,fname):#Method to persist corpus to disk, important as instantiation of object is computationally expensive
        if isinstance(directory,str) and isinstance(fname,str):
            if os.path.isdir(directory):
                if directory[-1]!="\\" and directory[-1]!=["/"]:
                    directory+="\\"
                with open(directory+fname+".pkl","wb") as dumpLocation:#Export using pickle
                    pickle.dump(self,dumpLocation)
                print("Successfully exported to" + directory + fname + ".pkl")#Save with .pkl to indicate that file is saved via pickle
            else:
                print("Directory does not exist")
        else:
            print("Invalid directory/fname. Both must be of type \"str\"")

if __name__=="__main__":
    corpusJSONPath="C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\CLEF\\FastText\\corpusVectors.json"
    saveDirectory="C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\CLEF\\FastText"
    fileName="CLEF-FastText"

    testCorpus=corpus(corpusJSON=corpusJSONPath)
    testCorpus.export(saveDirectory,fileName)
    # testCorpus=pickle.load(open("C:\\Users\\Intern-RM1V\\Desktop\\Clovis\\Project\\Corpus\\CLEF\\FastText\\CLEF-FastText.pkl","rb"))
    # print(testCorpus.documents[0])

