import math
import pandas as pd


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
            prediction = self.process(answer["student"], answer["item"], answer["correct"], answer)
            if self.logger is not None:
                self.logger(answer, prediction)
            
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


class AvgItemModel(Model):
    def __init__(self):
        Model.__init__(self)
        self.corrects = 0
        self.all = 0

    def __str__(self):
        return "Item average"

    def pre_process_data(self, data):
        items = data.get_items()
        self.corrects = pd.Series(index=items)
        self.counts = pd.Series(index=items)
        self.corrects.fill(0)
        self.counts.fill(0)

    def process(self, student, item, correct, extra=None):
        ret = self.corrects[item] / self.counts[item] if self.counts[item] > 0 else 0.7
        self.counts[item] += 1
        if correct:
            self.corrects[item] += 1

        return ret


