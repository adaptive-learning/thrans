from model import Model, sigmoid
import pandas as pd
from hashlib import sha1

class EloCorrModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, decay_function=None, corr_place_weight=1, prior_weight=0, place_decay=False, min_corr=None):
        Model.__init__(self)

        self.corr = None

        self.alpha = alpha
        self.beta = beta
        self.place_decay = place_decay
        self.decay_function = decay_function if decay_function is not None else lambda x: alpha / (1 + beta * x)
        self.corr_place_weight = corr_place_weight
        self.prior_weight = prior_weight
        self.min_corr = min_corr

        self.global_skill = {}
        self.difficulty = {}
        self.student_attempts = {}
        self.place_attempts = {}
        self.local_skill = {}

    def __str__(self):
        return "Elo with correlations{}; decay - alpha: {}, beta: {}, prior_weight {}, corr_place_weight {}{}"\
            .format(" plDe" if self.place_decay else "", self.alpha, self.beta, self.prior_weight, self.corr_place_weight, "" if self.min_corr is None else " min_corr: " + str(self.min_corr))

    def pre_process_data(self, data):
        filename = "data/{}{}.corr.pd".format(sha1(str(data)).hexdigest()[:10], "" if self.min_corr is None else " min_corr: " + str(self.min_corr))
        try:
            self.corr = pd.read_pickle(filename)
            return
        except:
            pass

        train_data = data.get_dataframe() if data.only_train else data.get_train_dataframe()

        print "Computing response matrix"
        responses = pd.DataFrame(index=train_data["student"].unique(), columns=train_data["item"].unique())
        answers = data if data.only_train else data.train_iter()
        for answer in answers:
            guess = 1. / answer["choices"] if answer["choices"] else 0
            responses.ix[answer["student"], answer["item"]] = answer["correct"] * 1 - guess

        print "Computing correlations"
        responses = responses.astype(float)
        self.corr = responses.corr(method="spearman", min_periods=self.min_corr)
        self.corr.fillna(0, inplace=True)
        print "NaNs in correlation matrix", (self.corr==0).sum()
        self.corr.to_pickle(filename)



    def initialize_if_needed(self, student, item):
        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.local_skill[student] = pd.Series([0]*len(self.corr.index), index=self.corr.index, dtype=float)
            self.student_attempts[student] = 0
        if not item in self.difficulty:
            self.difficulty[item] = 0
            self.place_attempts[item] = 0

    def process(self, student, item, correct, extra=None):
        self.initialize_if_needed(student, item)
        random_factor = 0 if extra is None or extra["choices"] == 0 else 1. / extra["choices"]
        if item not in self.local_skill[student]:
            print "unknown place", item
            return 0.7
        skill = self.corr_place_weight * self.local_skill[student][item] + self.prior_weight * self.global_skill[student]

        prediction = sigmoid(skill - self.difficulty[item], random_factor)
        dif = (correct - prediction)

        decay = self.decay_function(self.student_attempts[student]) if self.place_decay else 1
        self.local_skill[student] += dif * self.corr[item] * decay

        self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
        self.difficulty[item] -= self.decay_function(self.place_attempts[item]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[item] += 1
        return prediction
