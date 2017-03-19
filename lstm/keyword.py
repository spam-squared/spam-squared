import requests
import json

class Keyword(object):

    url = "http://api.meaningcloud.com/topics-2.0"

    def getKeywords(self, text):
        payload = "key=ca2aff042ede68ee24c7225387167015&lang=eng&txt="+text+"&tt=a"
        headers = {'content-type': 'application/x-www-form-urlencoded'}

        response = requests.request("POST", self.url, data=payload, headers=headers)

        data = json.loads(response)
        map(lambda x: x['form'], filter(lambda x: x['relevance'] >= 50,data['entity_list']))