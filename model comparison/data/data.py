import json
import pandas as pd


class Data():
    def __init__(self, filename, test=False):
        self.file = filename
        self.test = test
        self.n = -1         # not counted yet

    def __str__(self):
        if self.test:
            return "Test: " + self.file
        return self.file

    def __iter__(self):
        data = pd.load(self.file)
        self.n = len(data)

        columns = data.columns.values
        for row in data.values:
            yield dict(zip(columns, row))