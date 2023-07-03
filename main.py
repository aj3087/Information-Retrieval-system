from googleSearchModule import returnGoogleSearchResults
from metricsModule import returnPrecisionK
from queryExapansionModule import queryExpansion
import nltk
import sys
from urllib.parse import urlparse

def printResult(doc):
        '''
        Pretty printing of the individual search results.
        '''
        print("URL:",doc.url+"\nTitle:",doc.title,"\nSummary:",doc.summary)

def main(apiKey,engineId,desiredPrec,query):
    
    # main Entry point into the application
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

    try:
        
        print("Welcome to the application")
        print("API Key:",apiKey)
        print("Engine ID",engineId)
        print("Precision:",desiredPrec)

        #query preprocessing
        query = query.strip().lower()
        # Creating state varaiables to store relevent documents
        releventDoc = [] 
        iter = 1
        while True:
            print("Iteration No:",iter)
            print("Query:",query)
            
            relevenceJudgement = []
            
            # Fetching search results based on the original query
            results = returnGoogleSearchResults(query,apiKey,engineId)
                  
            for doc in results:
                # obtaining user feedback for the results
                printResult(doc)
                print("Is it relevant[Y/N]:",end="")
                rel = input().strip()
                
                # Fetching Relevenace feedback for the search Result
                if rel == "Y":
                    relevenceJudgement.append(1)
                    
                    # Extarcting the Domain name from the URl
                    urlDomain = urlparse(doc.url).netloc

                    # Adding the domain, title, summary to the in memeory relevantDoc store
                    releventDoc.append(urlDomain+" "+doc.title+" "+doc.summary)
                else:
                    relevenceJudgement.append(0)
                print()

            # Computing the Precision@k metrics for the result
            prec = returnPrecisionK(relevenceJudgement)
            print("Precision@10:",prec)
            
            # Evaluating if we acheieved the required Precision
            if prec >= float(desiredPrec):
                print("Desired Precision Met Exiting")
                break
            
            # Using Query Expansion to create a new augmented query for next iter
            query = queryExpansion(query,releventDoc)
            iter+=1
            
            print("---------------------------------------------------------------------------")
            print("---------------------------------------------------------------------------")

    except:
        print("Internal Server Error")
        
data=sys.argv[1:]
main(*data)