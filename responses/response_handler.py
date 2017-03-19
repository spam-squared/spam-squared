import json

responses = None


def load_responses():
    """Load responses from json and initialize them"""
    with open('responses.json') as data_file:
        data = json.load(data_file)
        global responses
        responses = data


def get_responses():
    """Get responses from response pool"""
    global responses
    return responses # Get with responses["responses"][index]["text"]
