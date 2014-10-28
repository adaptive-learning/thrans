import json
import math


def sigmoid(x, c=0):
    return c + (1 - c) / (1 + math.exp(-x))


class Model:
    def __init__(self):
        self.logger = None

    def __str__(self):
        return "Not specified"

    def pre_process_data(self, data):
        pass

    def process_data(self, data):
        for answer in data:
            prediction = self.process(answer["student"], answer["item"], answer["correct"], answer["extra"])
            answer["prediction"] = prediction
            if self.logger is not None:
                self.logger(json.dumps(answer))
            
    def process(self, student, item, correct, extra=None):
        pass


class AvgModel(Model):
    def __init__(self):
        Model.__init__(self)
        self.corrects = 0
        self.all = 0

        self.avg = None

    def __str__(self):
        return "Global average"

    def pre_process_data(self, data):
        for answer in data:
            self.all += 1
            if answer["correct"]:
                self.corrects += 1
        self.avg = float(self.corrects) / self.all

    def process(self, student, item, correct, extra=None):
        return self.avg


