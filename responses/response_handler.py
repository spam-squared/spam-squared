import json


class ResponseHandler(object):
    responses = []

    def load_responses(self):
        """Load responses from json and initialize them"""
        with open('responses.json') as data_file:
            data = json.load(data_file)
            self.responses = data

    def get_responses(self):
        """Get responses from response pool"""
        return self.responses # Get with responses["responses"][index]["text"]
