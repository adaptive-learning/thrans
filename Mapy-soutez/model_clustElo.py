import json
import random
import models
import math
from data import Data
from evaluator import *
import pandas as pd


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_shift(x, c):
    return c + (1-c) * sigmoid(x)


def random_factor(qtype):
    if qtype == 10: return 0
    return 1.0 / (qtype - 10*(qtype//10))


class ModelElo_clust(models.ModelElo):
    def __init__(self, alpha=1.0, beta=0.05, ufun=None, ):
        models.ModelElo.__init__(self)
        self.__dict__.update(locals())
        if ufun is None:
            self.ufun = lambda x: alpha / (1 + beta * x)
        else:
            self.ufun = ufun
        self.global_skill = {}
        self.difficulty = {}
        self.student_attempts = {}
        self.place_attempts = {}

        self.maps = json.load(open("maps.json"))
        self.maps_map = {}
        for map, places in self.maps.items():
            for place in places:
                self.maps_map[place] = map

    def __str__(self):
        return "Elo " + `self.alpha`+" "+`self.beta`

    def initialize_if_needed(self, student, place):
        if not student in self.global_skill:
            self.global_skill[student] = {}
            self.student_attempts[student] = 0
        map = self.maps_map[place]
        if not map in self.global_skill[student]:
            self.global_skill[student][map] = 0
        if not place in self.difficulty:
            self.difficulty[place] = 0
            self.place_attempts[place] = 0

    def process(self, student, place, correct, qtype):
        map = self.maps_map[place]
        self.initialize_if_needed(student, place)
        pred = sigmoid_shift(self.global_skill[student][map] - self.difficulty[place], random_factor(qtype))
        dif = (correct - pred)
        self.global_skill[student][map] +=  self.ufun(self.student_attempts[student]) * dif
        self.difficulty[place] -= self.ufun(self.place_attempts[place]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[place] += 1
        return pred


class ModelElo_clust2(models.ModelElo):
    def __init__(self, alpha=0.4, beta=0.05, corr_places_count=0, corr_place_weight=1, prior_weight=0):
        models.ModelElo.__init__(self, alpha, beta)

        self.most_correlated_places = {}
        self.local_skill = {}

        self.corr_place_weight = corr_place_weight
        self.corr_places_count = corr_places_count
        self.prior_weight = prior_weight

        self.maps = json.load(open("maps2.json"))
        self.maps_map = {}
        for map, places in self.maps.items():
            for place in places:
                self.maps_map[place] = map

        self.corr = pd.read_pickle("clust_corr.pd")
        self.corr[self.corr == 1.] = 0
        self.corr[self.corr.isnull()] = 0
        for place, c in self.corr.iteritems():
            cs = c.copy()
            cs.sort(ascending=False)
            self.most_correlated_places[place] = cs.index

    def __str__(self):
        return "Elo with correlations, corr_places_count {0}, corr_place_weight {3}, alpha: {1}, beta{2}".format(self.corr_places_count, self.alpha, self.beta, self.corr_place_weight)

    def initialize_if_needed(self, student, place):
        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.local_skill[student] = pd.Series([0]*len(self.corr.index), index=self.corr.index, dtype=float)
            self.student_attempts[student] = 0
        if not place in self.difficulty:
            self.difficulty[place] = 0
            self.place_attempts[place] = 0

    def process(self, student, place, correct, qtype):
        self.initialize_if_needed(student, place)
        map = self.maps_map[place]
        skill = self.corr_place_weight * self.local_skill[student][map] + self.prior_weight * self.global_skill[student]

        pred = sigmoid_shift(skill - self.difficulty[place], random_factor(qtype))
        dif = (correct - pred)

        self.local_skill[student] += dif * self.corr[map]

        self.global_skill[student] += self.ufun(self.student_attempts[student]) * dif
        self.difficulty[place] -= self.ufun(self.place_attempts[place]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[place] += 1
        return pred


if False:
    data1 = Data("data-first-0.5-1.csv", places=None)
    data2 = Data("data-first-0.5-2.csv", places=None)

if False:
    places = []
    maps = json.load(open("maps.json"))
    for _, ps in maps.items():
        places += ps
    data = Data("data-first-10000.csv", places=places)

    m = ModelElo_clust2()
    m.process_data(data)
    e = Evaluator(m.log)
    e.print_metrics()
    e.calibration_graphs()
    plt.show()



if False:
    places = []
    maps = json.load(open("maps.json"))
    for _, ps in maps.items():
        places += ps
    data = Data("data-first.csv", places=places)

    m = ModelElo_clust()
    m.process_data(data)

    skills = pd.DataFrame(columns=maps.keys(), index=data.users)
    for student, ss in m.global_skill.items():
        for map, s in ss.items():
            skills[map][student] = s

    skills = skills.astype(float)
    skills.corr(method="spearman").to_pickle("clust_corr.pd")
    # e = Evaluator(m.log)
    # e.print_metrics()
    # e.calibration_graphs()
    # plt.show()
















