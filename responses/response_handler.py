import json
from random import randint


class ResponseHandler(object):
    responses = []

    def load_responses(self):
        """Load responses from json and initialize them"""
        with open('./responses/responses.json') as data_file:
            data = json.load(data_file)
            self.responses = data

    def get_responses(self):
        """Get responses from response pool"""
        return self.responses # Get with responses["responses"][index]["text"]

    def get_random_index(self):
        return randint(0, len(self.responses))
