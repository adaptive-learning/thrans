from model import Model, sigmoid


class EloModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, decay_function=None, gamma=None):
        Model.__init__(self)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if gamma is None:
            self.decay_function = decay_function if decay_function is not None else lambda x: alpha / (1 + beta * x)
        else:
            self.decay_function = decay_function if decay_function is not None else lambda x: gamma + alpha / (1 + beta * x)

        self.global_skill = {}
        self.difficulty = {}
        self.student_attempts = {}
        self.place_attempts = {}

    def __str__(self):
        if self.gamma is None:
            return "Elo; decay - alpha: {}, beta: {}".format(self.alpha, self.beta)
        else:
            return "Elo; decay - alpha: {}, beta: {}, gamma: {}".format(self.alpha, self.beta, self.gamma)

    def initialize_if_needed(self, student, item):
        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.student_attempts[student] = 0
        if not item in self.difficulty:
            self.difficulty[item] = 0
            self.place_attempts[item] = 0

    def process(self, student, item, correct, extra=None):
        self.initialize_if_needed(student, item)
        random_factor = 0 if extra is None or extra["choices"] == 0 else 1. / extra["choices"]

        prediction = sigmoid(self.global_skill[student] - self.difficulty[item], random_factor)
        dif = (correct - prediction)

        self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
        self.difficulty[item] -= self.decay_function(self.place_attempts[item]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[item] += 1

        return prediction