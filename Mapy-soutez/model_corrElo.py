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


def compute_corr_matrix(data, users_req=200, answ_req=50):
    try:
        return pd.read_pickle("corrs/2"+str(data)+".pd")
    except:
        pass

    results = pd.DataFrame(index=data.users, columns=data.places)
    for i in range(data.n):
        user = data.student[i]
        place = data.place[i]
        correct = data.correct[i]
        rand = random_factor(data.qtype[i])
        if np.isnan(results.ix[user, place]):
            results.ix[user, place] = correct - rand

    results = results.ix[results.count(1) > answ_req, results.count() > 0]
    filter = results.count() < users_req
    print "place zeroed", sum(filter)
    results = results.astype(float)

    corr = results.corr(method="spearman")
    corr.ix[filter, :] = 0
    corr.ix[:, filter] = 0

    corr.to_pickle("corrs/2"+str(data)+".pd")
    return corr


class ModelCorrElo(models.ModelElo):
    def __init__(self, data, alpha=0.4, beta=0.05, places=None, corr_places_count=0, corr_place_weight=2.5, prior_weight=0):
        models.ModelElo.__init__(self, alpha, beta)

        self.most_correlated_places = {}
        self.local_skill = {}

        self.corr_place_weight = corr_place_weight
        self.corr_places_count = corr_places_count
        self.prior_weight = prior_weight

        self.corr = compute_corr_matrix(data)
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
        # if self.corr_places_count == 0:
        #     correction = self.difs[student].dot(self.corr[place])
        # else:
        #     correction = 0
        #     for mcp in self.most_correlated_places[place][:self.corr_places_count]:
        #         if mcp in self.local_data[student] and mcp != place:
        #             correction += self.corr.loc[place, mcp] * self.local_data[student][mcp]["diff"]

        # skill = self.prior_weight * self.global_skill[student] + self.corr_place_weight * correction
        skill = self.corr_place_weight * self.local_skill[student][place] + self.prior_weight * self.global_skill[student]

        pred = sigmoid_shift(skill - self.difficulty[place], random_factor(qtype))
        dif = (correct - pred)

        self.local_skill[student] += dif * self.corr[place]

        self.global_skill[student] += self.ufun(self.student_attempts[student]) * dif
        self.difficulty[place] -= self.ufun(self.place_attempts[place]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[place] += 1
        return pred


europe = [51, 64, 66, 70, 74, 78, 79, 81, 88, 93, 94, 108, 113, 114, 115, 142, 143, 144, 146, 147, 154, 159, 164, 165, 176, 178, 179, 181, 182, 184, 190, 191, 194, 196, 203, 205, 206, 216, 234]

if False:
    data1 = Data("data-first-0.5-1.csv", places=None)
    data2 = Data("data-first-0.5-2.csv", places=None)
    corr1 = compute_corr_matrix(data1)
    # print corr1
    corr2 = compute_corr_matrix(data2)

    place = 56
    print corr2[place].corr(corr1[place], method="spearman")
    print corr2[place].corr(corr1[place])
    plt.plot(corr2[place], corr1[place], "o")
    plt.show()

if False:
    data = Data("data-first-10000.csv")

    m = ModelCorrElo(data)
    m.process_data(data)
    e = Evaluator(m.log)
    e.print_metrics()
    e.calibration_graphs()
    plt.show()
