'''
This is designed for the new Azure Marketplace Bing Search API (released Aug 2012)
Inspired by https://github.com/mlagace/Python-SimpleBing and
http://social.msdn.microsoft.com/Forums/pl-PL/windowsazuretroubleshooting/thread/450293bb-fa86-46ef-be7e-9c18dfb991ad
'''

import requests # Get from https://github.com/kennethreitz/requests
import string
from pdb import set_trace as trace
from sklearn.externals import joblib
from datetime import datetime
class BingSearchAPI():
    bing_api = "https://api.datamarket.azure.com/Data.ashx/Bing/Search/v1/Composite?"

    def __init__(self, key):
        self.key = key

    def replace_symbols(self, request):
        # Custom urlencoder.
        # They specifically want %27 as the quotation which is a single quote '
        # We're going to map both ' and " to %27 to make it more python-esque
        request = string.replace(request, "'", '%27')
        request = string.replace(request, '"', '%27')
        request = string.replace(request, '+', '%2b')
        request = string.replace(request, ' ', '%20')
        request = string.replace(request, ':', '%3a')
        return request

    def search(self, sources, query, params):
        ''' This function expects a dictionary of query parameters and values.
            Sources and Query are mandatory fields.
            Sources is required to be the first parameter.
            Both Sources and Query requires single quotes surrounding it.
            All parameters are case sensitive. Go figure.
            For the Bing Search API schema, go to http://www.bing.com/developers/
            Click on Bing Search API. Then download the Bing API Schema Guide
            (which is oddly a word document file...pretty lame for a web api doc)
        '''
        request =  'Sources="' + sources    + '"'
        request += '&Query="'  + str(query) + '"'
        for key,value in params.iteritems():
            request += '&' + key + '=' + str(value)
        request = self.bing_api + self.replace_symbols(request)
        return requests.get(request, auth=(self.key, self.key))


if __name__ == "__main__":
    start = datetime.now()
    my_key = "-my-key-"
    #query_string = "Microsoft"
    dataset = joblib.load('test_dataset.joblib')
    dataset = dataset.fillna('')
    print 'Data Loading is Completed', dataset.shape
    cmpy_dict = {}
    counter=0
    cmpy_lst = []
    for index, row in dataset.iterrows():
        company = row['Companies']
        if company=='':
            continue
        elif list(set(company)) in ([], ['']):
            continue
        else:
            for cmpy in company:
                if cmpy=='':
                    continue
                elif list(set(cmpy)) in ([], ['']):
                    continue
                cmpy_lst.append(cmpy)

    for company in cmpy_lst:
        query_string = company
        bing = BingSearchAPI(my_key)
        params = {'ImageFilters':'"Face:Face"', '$format': 'json', '$top': 10, '$skip': 0}
        try:
            temp = bing.search('image+web', query_string, params).json()
            cmpy_dict[company]=str(temp.get('d').get('results')[0].get('Web')[0].get('Description'))
        except:
            pass

    print datetime.now() - start
    trace()
