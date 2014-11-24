from collections import defaultdict
import json
from proso.geography import places, answers
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


def find_maps():
    types = {
        1: "country",
        2: "city",
        3: "world",
        4: "continent",
        5: "river",
        6: "lake",
        7: "region-(cz)",
        8: "bundesland",
        9: "province",
        10: "region-(it)",
        11: "region",
        12: "autonomus-community",
        13: "mountains",
        14: "island",
    }

    # answers_all = answers.from_csv("raw data/geography-all.csv")
    maps = defaultdict(lambda: [])
    places_all = places.from_csv("raw data/geography.place.csv", "raw data/geography.placerelation.csv","raw data/geography.placerelation_related_places.csv")
    places_all.set_index(places_all["id"], inplace=True)
    for _, place in places_all.iterrows():
        for id, relation in place.relations:
            if relation == "is_on_map":
                key = places_all.ix[id]["name"]+"-"+types[place.type]
                maps[key].append(place.id)

    with open("maps.json", "w") as f:
        json.dump(maps, f)


def get_maps():
    with open("maps.json") as f:
        return json.load(f)


# maps = get_maps()
# places_all = places.from_csv("raw data/geography.place.csv", "raw data/geography.placerelation.csv","raw data/geography.placerelation_related_places.csv")
# places_all.set_index(places_all["id"], inplace=True)
# for p in maps["Czech Rep.-city"]:
#     print places_all.ix[p]
