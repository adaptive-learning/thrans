import json
import random
import pandas as pd


class Data():
    def __init__(self, filename, test=False, train=None):
        self.file = filename
        self.test = test
        self.n = -1         # not counted yet
        self.data = None
        self.data_train = None
        self.train = train

    def __str__(self):
        if self.test:
            return "Test: " + self.file
        if self.train:
            return self.file + " - train: {}".format(self.train)
        return self.file

    def get_dataframe(self):
        self.load_file()
        return self.data

    def get_train_dataframe(self):
        self.load_file()
        return self.data_train


    def load_file(self):
        if self.data is None:
            self.data = pd.load(self.file)

            if self.train is not None:
                random.seed(42)
                students = self.get_students()
                selected_students = random.sample(students, int(len(students) * self.train))
                self.data_train = self.data[self.data["student"].isin(selected_students)]
                self.data = self.data[~self.data["student"].isin(selected_students)]

            self.n = len(self.data)
            if self.test:
                self.data = self.data[:10000]

    def join_predictions(self, predictions):
        self.load_file()
        self.data["prediction"] = predictions

    def get_items(self):
        self.load_file()
        return self.data["item"].unique()

    def get_students(self):
        self.load_file()
        return self.data["student"].unique()

    def __iter__(self):
        self.load_file()

        columns = self.data.columns.values
        i = 0
        for row in self.data.values:
            i += 1
            yield dict(zip(columns, row))
            if i % 10000 == 0:
               print ".",
        print

    def train_iter(self):
        self.load_file()

        columns = self.data_train.columns.values
        i = 0
        for row in self.data_train.values:
            i += 1
            yield dict(zip(columns, row))
            if i % 10000 == 0:
                print ".",
        print