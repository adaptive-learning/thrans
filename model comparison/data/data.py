from collections import defaultdict
import json
import os
from proso.geography import places, answers
import random
import pandas as pd


class Data():
    def __init__(self, filename, test=False, train=None, force_train=False, only_train=False):
        self.file = filename
        self.test = test
        self.n = -1         # not counted yet
        self.data = None
        self.data_train = None
        self.train = train
        self.force_train = force_train
        self.only_train = only_train

    def __str__(self):
        if self.test:
            return "Test: " + self.file
        if self.train:
            return self.file + " - train{1}: {0}{2}".format(self.train, " only" if self.only_train else "", " seed:"+str(self.force_train) if self.force_train else "")
        return self.file

    def get_dataframe(self):
        self.load_file()
        return self.data

    def get_train_dataframe(self):
        self.load_file()
        return self.data_train

    def get_dataframe_all(self):
        self.load_file()
        return self.all_data

    def load_file(self):
        if self.data is None:
            self.data = pd.read_pickle(self.file)

            if self.train is not None:
                self.all_data = self.data
                if self.force_train:
                    random.seed(self.force_train)
                    students = self.get_students()
                    selected_students = random.sample(students, int(len(students) * self.train))
                else:
                    selected_students = json.load(open(os.path.dirname(os.path.abspath(__file__)) + "/train_users.json"))
                self.data_train = self.data[self.data["student"].isin(selected_students)]
                self.data = self.data[~self.data["student"].isin(selected_students)]
            else:
                self.data_train = self.data[self.data["student"].isin([])]

            if self.only_train:
                self.all_data = self.data_train
                self.data = self.data_train
                self.data_train = pd.DataFrame(columns=self.all_data.columns)

            if self.test:
                self.data = self.data[:10000]
            self.n = len(self.data)

    def join_predictions(self, predictions):
        self.load_file()
        self.data["prediction"] = predictions

    def get_items(self):
        self.load_file()
        return self.all_data["item"].unique()

    def get_students(self):
        self.load_file()
        return self.all_data["student"].unique()

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

    def all_iter(self):
        self.load_file()

        columns = self.all_data.columns.values
        i = 0
        for row in self.all_data.values:
            i += 1
            yield dict(zip(columns, row))
            if i % 10000 == 0:
                print ".",
        print