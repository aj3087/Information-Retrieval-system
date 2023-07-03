from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
from nltk import pos_tag
import math

def findProximity(word1,word2,corpus):
    '''
    Given terms word1, word2 , corpus
    Returns: (minDistance between word1 and word2,director either 1/-1)
    '''

    diff =  float('inf')
    direc = 1
    index1List = []
    index2List = []

    #populating index1List and index2List
    for index,term in enumerate(corpus):
        if term == word1:
            index1List.append(index)
        elif term == word2:
            index2List.append(index)

    n,m = len(index1List), len(index2List)
    i,j= 0,0

    while i<n and j<m:
        if diff > abs(index1List[i]-index2List[j]):
            diff = abs(index1List[i]-index2List[j])
            direc = 1 if index1List[i] > index2List[j] else -1

        if diff == 1:
            return (diff,direc) # this is the minimum value possible
        if index1List[i]< index2List[j]:
            i+=1
        else:
            j+=1

    return (diff,direc)      

def nGramBasedQE(tokenizedQuery,tokenizedCorpus,n):
    '''
    takes tokenizedQuery, tokenizedCorpus , n
    n : ngrams
    returns: (max ngram Score,newQuery)
    '''
    # Computing ngrams in the query and corpus
    queryNgrams = list(ngrams(tokenizedQuery,n))
    corpusNgrams = list(ngrams(tokenizedCorpus,n))

    # Computing the raw freq of ngrams in the corpus
    ngramFreq = Counter(corpusNgrams)

    queryLen = len(queryNgrams)

    proximityScore = [{} for _ in range(queryLen)]

    for term in ngramFreq:
        for index,queryToken in enumerate(queryNgrams):
            
            # Discarding a ngram if it shares a common term with the tojenized query 
            if term not in queryNgrams:
                flag = True
                for item in term:
                    if item in tokenizedQuery:
                        flag = False
                        break
                if flag:
                    # Computing the proximity score between ngram and the queryToken
                    proximityScore[index][term] = findProximity(term,queryToken,corpusNgrams)
    

    maxScore,maxTerm,maxIndex,maxDir = float('-inf'),'', queryLen,1
    scoreDict = {}
    alpha = 0.6

    for term in ngramFreq:
        if term not in queryNgrams:
            score1 = math.log10(ngramFreq[term])  # Collection frequency
            pos_tags = pos_tag(list(term)) 
            #In case any one of terms is a Noun we give it a higher priority. Hence collection frequency is scaled up by a factor of 2
            if any([True if ele[1]=="NN" else False for ele in pos_tags]):
                score1 *=2
                
            # We need to find the query term corresponsiding to which the given term has minimum proximity
            # Proximity refers to the distance between 2 words
            minProximity,minIndex,dir = float('inf'),-1,1
            for index in range(queryLen):
                if term in proximityScore[index] and proximityScore[index][term][0] < minProximity:
                    minProximity = proximityScore[index][term][0]
                    minIndex = index
                    dir = proximityScore[index][term][1]
            
            if minProximity != float('inf'):
                # Second score 1/minProximity since score is inversely propoitonal to distance/proxmimty between tokens 
                score2=math.log10(1/minProximity)
            else:
                # No proximity hence does not exists
                continue

            
            scoreDict[term] = [score1,score2]

            # identifying term with highest score
            if alpha*score1 + (1-alpha)*score2 >= maxScore:
                maxScore = alpha*score1 + (1-alpha)*score2
                maxTerm = term
                maxIndex = minIndex
                maxDir = dir
    
    # Appending the term with highest score in the query based on the index of minProximity query term and direction
    if maxDir == 1:
        tokenizedQuery = tokenizedQuery[:maxIndex+2]+list(maxTerm)+tokenizedQuery[maxIndex+2:]
    else:
        tokenizedQuery = tokenizedQuery[:maxIndex-2]+list(maxTerm)+tokenizedQuery[maxIndex-2:]


    return (maxScore,' '.join(tokenizedQuery))



def queryExpansion(query,relevantDocument):
    '''
    Accepts: Query , relevanrDocuments
    returns: newQuery
    
    '''
    
    # Tokenize the individual documents
    stopWords = list(stopwords.words('english'))+['...','\xa0','.','-','|',',']

    docFreq= {}
    rawTermFreq = Counter([])
    netCorpus = []

    # Computing the document frequenct and collection Freq
    for doc in relevantDocument:
        tokenizedDocument = list(word_tokenize(doc)) # tokenize the document
        docCounter = Counter(tokenizedDocument)  # determine raw term freq
        rawTermFreq = rawTermFreq + docCounter # maintain the collection frequency

        netCorpus = netCorpus + tokenizedDocument # builing the netCorpus

        for term in docCounter:
            docFreq[term] = docFreq.get(term,0)+1
        
    netDocs = len(relevantDocument)  # The net no of documents
    
    # removing the Stop words from the corpus
    for removeTerm in stopWords:
        while removeTerm in netCorpus:
            netCorpus.remove(removeTerm)
    
    # Processing the query

    tokenizedQuery = list(word_tokenize(query)) # Tokenizing the query
    queryLen = len(tokenizedQuery)
    proximityScore = [{}]*queryLen

    # Computing proximity score for each term and query term Combination
    for term in rawTermFreq:

        for index,queryToken in enumerate(tokenizedQuery):
            # Ensuring the newterm is not a stopword and is not present in the query already
            if queryToken!=term and term not in stopWords and queryToken in rawTermFreq:
                proximityScore[index][term] = findProximity(term,queryToken,netCorpus)

    # calculating the final scores for each of the words in the document
    maxScore,maxTerm,maxIndex,maxDir = float('-inf'),'', queryLen,1
    
    # Hyper parameter alpha
    alpha = 0.5
    beta = 0.3
    scoreDict = {}

    for term in rawTermFreq:
        if term not in tokenizedQuery and term not in stopWords:
            #We will check POS of the term if it has a noun we will increase it's frequency by 2 to give it more priority over other items.
            pos_tags = pos_tag([term])[0][1]
            score1 = math.log10(rawTermFreq[term]) # Collection frequency
            if pos_tags == "NN":
                score1 *= 2                  

            score3 = math.log10(netDocs/docFreq[term]) # idf of the given term

            # We need to find the query term corresponsiding to which the given term has minimum proximity
            minProximity,minIndex,dir = float('inf'),-1,1
            for index in range(queryLen):
                if term in proximityScore[index] and proximityScore[index][term][0] < minProximity:
                    minProximity = proximityScore[index][term][0]
                    minIndex = index
                    dir = proximityScore[index][term][1]
            
            if minProximity !=float('inf'):
                # Computing the score which is inversly propotional to proximity
                score2 = math.log10(1/minProximity)
            else:
                score2 = 0
            
            scoreDict[term] = [score1,score2,score3]

            # Computing the unigram Scores
            if alpha*score1 + (beta)*score2 + (1-alpha-beta)*score3 > maxScore:
                maxScore = alpha*score1 + (beta)*score2 + (1-alpha-beta)*score3
                maxTerm = term
                maxIndex = minIndex
                maxDir = dir

    # biGram scores are computed only if two query terms
    if queryLen >=2:
        bigramScore,bigramQuery = nGramBasedQE(tokenizedQuery,netCorpus,2) # bigram score
    else:
        bigramScore = 0

    if bigramScore*5 > maxScore: # Scaling up bigram Scores before comapring it with unigram scores
        return bigramQuery # return bigram Query if it is larger than uniGram Scores
    
    # One Issue is is the order of insertion should be before the maxElement or after the maxElement
    if maxDir == 1:
        tokenizedQuery.insert(maxIndex+1,maxTerm)
    else:
        tokenizedQuery.insert(maxIndex-1,maxTerm)

    # If Unigram score is higher return this
    return ' '.join(tokenizedQuery)


# remember to dowload nltk.download("punkt")
#print(queryExpansion("page",["larry page  wikipedia lawrence edward page born march 26 1973 is an american business magnate computer scientist and internet entrepreneur he is best known for cofounding"]))



    





