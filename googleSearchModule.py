import string
import requests
import json
class searchResult:
    '''
    Class that preprocesses and houses the google search results.
    '''
    def __init__(self,url,title,summary):
        self.url = url
        if title:
            self.title = title.lower().translate(str.maketrans('','',string.punctuation))
        else:
            self.title = ""
        if summary:
            self.summary = summary.lower().translate(str.maketrans('','',string.punctuation))       
        else:
            self.summary = ""


def returnGoogleSearchResults(query,apiKey='',engineId=''):
    '''
    Function takes in the query , Api key and engine Id and fetches the top 10 results from the google search results.
    '''
    # This mechanism is to use the config file for apiKey and engineId in case they are missing from the parameters
    with open('./Config.json', 'r') as myfile:
        data=myfile.read()
    config =  json.loads(data)
    if apiKey=='':
        apiKey=config['API_KEY']
    if engineId=='':
        engineId=config['SEARCH_ENGINE_ID']
    start = 1

    res = []  

    '''
        Making a call to google custom search API 
    '''
    
    try:
        url = "https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}".format(API_KEY=apiKey,SEARCH_ENGINE_ID=engineId,query=query)
        results = requests.get(url).json()
    except:
        raise Exception("Unexpected Error while making the search engine call.")

    if results == None:
        raise Exception("No results returned")

    search_results = results.get("items")
    

    for i, search_item in enumerate(search_results):
        '''
            Processing the results returned by google search results
        '''

        try:
            fileType = search_item["fileFormat"]
        except KeyError:
            fileType = "NA"

        if fileType != "NA":
            # it is a non html file so not including them
            continue
        
        try:
            title = search_item.get("title")
        except KeyError:
            title = ""
        
        
        try:
            snippet = search_item.get("snippet")
        except KeyError:
            snippet = ""

        try:
            link = search_item.get("link")
        except KeyError:
            link = ""
        
        # extract the page url
        # print the results
        item = searchResult(link,title,snippet)

        # add search result to the document list returned to user
        res.append(item)
    
    return res

#pprint.pprint(returnGoogleSearchResults("pikachu"))


    
