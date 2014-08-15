import json
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


class Model:

    def __init__(self):
        pass

    def process_data(self, data):
        self.log = []
        for i in range(data.n):
            pred = self.process(data.student[i], data.place[i], data.correct[i], data.qtype[i])
            self.log.append((pred, data.correct[i]))
            
    def process(self, student, place, correct, qtype):
        pass    


class ModelElo(Model):

    def __init__(self, alpha = 1.0, beta = 0.05, ufun=None):
        self.__dict__.update(locals())
        if ufun is None:
            self.ufun = lambda x: alpha / (1 + beta * x)
        else:
            self.ufun = ufun
        
        self.global_skill = {}
        self.difficulty = {}
        self.student_attempts = {}
        self.place_attempts = {}

    def __str__(self):
        return "Elo " + `self.alpha`+" "+`self.beta`
        
    def initialize_if_needed(self, student, place):
        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.student_attempts[student] = 0 
        if not place in self.difficulty:
            self.difficulty[place] = 0
            self.place_attempts[place] = 0
            
    def process(self, student, place, correct, qtype):
        self.initialize_if_needed(student, place)
        pred = sigmoid_shift(self.global_skill[student] - self.difficulty[place], random_factor(qtype))
        dif = (correct - pred)
        self.global_skill[student] +=  self.ufun(self.student_attempts[student]) * dif
        self.difficulty[place] -= self.ufun(self.place_attempts[place]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[place] += 1
        return pred


class ModelStupid(Model):

    def __init__(self):
        self.count = 0
        self.sum  = 1

    def __str__(self):
        return "Constant prediction 0.8 "

    def process(self, student, place, correct, qtype):
        self.count += 1
        self.sum += correct

        return 0.745


class ModelEloCorr(ModelElo):
    def __init__(self, alpha=1.0, beta=0.05, places=None, corr_places_count=5, corr_place_weight=1, prior_weight=1):
        ModelElo.__init__(self, alpha, beta)

        self.most_correlated_places = {}
        self.local_data = {}
        self.difs = {}

        self.corr_place_weight = corr_place_weight
        self.corr_places_count = corr_places_count
        self.prior_weight = prior_weight

        corr = pd.read_pickle("correlations.pd")
        places = set(places) & set(corr.index)
        self.corr = corr.loc[places, places]
        self.corr[self.corr == 1.] = 0
        self.corr[self.corr.isnull()] = 0
        for place, c in self.corr.iteritems():
            cs = c.copy()
            cs.sort(ascending=False)
            self.most_correlated_places[place] = cs.index

    def __str__(self):
        return "Elo with correlations, corr_places_count {0}, corr_place_weight {3}, alpha: {1}, beta{2}".format(self.corr_places_count, self.alpha, self.beta, self.corr_place_weight)

    def ufun2(self, correct):
        if correct:
            return 3
        return -0.5

    def initialize_if_needed(self, student, place):
        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.local_data[student] = {}
            self.difs[student] = pd.Series([0]*len(self.corr.index), index=self.corr.index, dtype=float)
            self.student_attempts[student] = 0
        if not place in self.difficulty:
            self.difficulty[place] = 0
            self.place_attempts[place] = 0
        if not place in self.local_data[student]:
            self.local_data[student][place] = {
            }

    def process(self, student, place, correct, qtype):
        self.initialize_if_needed(student, place)
        if self.corr_places_count == 0:
            correction = self.difs[student].dot(self.corr[place])
        else:
            correction = 0
            for mcp in self.most_correlated_places[place][:self.corr_places_count]:
                if mcp in self.local_data[student] and mcp != place:
                    correction += self.corr.loc[place, mcp] * self.local_data[student][mcp]["diff"]

        skill = self.prior_weight * self.global_skill[student] + self.corr_place_weight * correction

        pred = sigmoid_shift(skill - self.difficulty[place], random_factor(qtype))
        dif = (correct - pred)

        self.local_data[student][place]["answer"] = correct
        self.local_data[student][place]["random_factor"] = random_factor(qtype)
        self.local_data[student][place]["diff"] = dif
        self.difs[student][place] = dif

        self.global_skill[student] += self.ufun(self.student_attempts[student]) * dif
        self.difficulty[place] -= self.ufun(self.place_attempts[place]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[place] += 1
        return pred


class ModelEloClust(ModelElo):
    def __init__(self, alpha=1.0, beta=0.05, clusters=None):
        ModelElo.__init__(self, alpha, beta)

        self.x = []

        self.most_correlated_places = {}
        self.local_skills = {}
        self.place_cluster_map = {}
        self.clusters = clusters
        for cluster, places in clusters.items():
            for p in places:
                self.place_cluster_map[p] = cluster

    def __str__(self):
        return "Elo with local skill for clusters" + str(self.clusters.keys()) + ", "  + `self.alpha`+" "+`self.beta`

    def initialize_if_needed(self, student, place):
        self.x.append((student, place))


        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.local_skills[student] = {}
            self.student_attempts[student] = {}
            self.student_attempts[student]["total"] = 0
        if not place in self.difficulty:
            self.difficulty[place] = 0
            self.place_attempts[place] = 0
        cluster = self.place_cluster_map[place]
        if not cluster in self.local_skills[student]:
            self.local_skills[student][cluster] = 0
            self.student_attempts[student][cluster] = 0

    def process(self, student, place, correct, qtype):
        self.initialize_if_needed(student, place)
        cluster = self.place_cluster_map[place]
        skill = self.global_skill[student] + self.local_skills[student][cluster]

        pred = sigmoid_shift(skill - self.difficulty[place], random_factor(qtype))
        dif = (correct - pred)

        self.global_skill[student] += self.ufun(self.student_attempts[student]["total"]) * dif
        self.local_skills[student][cluster] += self.ufun(self.student_attempts[student][cluster]) * dif
        self.difficulty[place] -= self.ufun(self.place_attempts[place]) * dif

        self.student_attempts[student]["total"] += 1
        self.student_attempts[student][cluster] += 1
        self.place_attempts[place] += 1

        return pred

if False:
    places = []
    with open("clusters_continents.json") as f:
        clusters = json.load(f)
        for name, ps in clusters.items():
            places += ps

    data = Data("data-first.csv", places=places)

    m = ModelEloClust(clusters=clusters)
    m.process_data(data)
    e = Evaluator(m.log)
    e.print_metrics()
    e.calibration_graphs()
    # plt.show()

if False:
    corr = pd.read_pickle("correlations.pd")
    data = Data("data-first-10000.csv", places=corr.index)

    m = ModelEloCorr(places=data.places, corr_places_count=0)
    m.process_data(data)
    e = Evaluator(m.log)
    e.print_metrics()
    e.calibration_graphs()
    plt.show()

if False:
    corr = pd.read_pickle("correlations.pd")
    data = Data("data-first.csv", places=corr.index)
    m = ModelStupid()
    m.process_data(data)
    print float(m.sum) / m.count