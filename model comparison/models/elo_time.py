import math
from model import Model, sigmoid


class EloTimeModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, decay_function=None, time_penalty_slope=0.9):
        Model.__init__(self)

        self.alpha = alpha
        self.beta = beta
        self.decay_function = decay_function if decay_function is not None else lambda x: alpha / (1 + beta * x)

        self.global_skill = {}
        self.difficulty = {}
        self.student_attempts = {}
        self.place_attempts = {}

        self.time_penalty_slope = time_penalty_slope
        self.attempts_count = 0
        self.time_intensity = 0

    def __str__(self):
        return "Elo time; decay - alpha: {}, beta: {}, slope: {}".format(self.alpha, self.beta, self.time_penalty_slope)

    def initialize_if_needed(self, student, item):
        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.student_attempts[student] = 0
        if not item in self.difficulty:
            self.difficulty[item] = 0
            self.place_attempts[item] = 0

    def get_response(self, solving_time):
        solving_time = min(solving_time / 1000., 100)
        expected_solving_time = math.exp(self.time_intensity)
        if expected_solving_time > solving_time:
            response = 1
        else:
            response = self.time_penalty_slope ** ((solving_time / expected_solving_time) - 1)
        return response

    def process(self, student, item, correct, extra=None):
        self.initialize_if_needed(student, item)
        random_factor = 0 if extra is None or extra["choices"] == 0 else 1. / extra["choices"]

        prediction = sigmoid(self.global_skill[student] - self.difficulty[item], random_factor)
        dif = (correct * self.get_response(extra["time"]) - prediction)

        self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
        self.difficulty[item] -= self.decay_function(self.place_attempts[item]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[item] += 1
        self.attempts_count += 1

        TI_delta = (math.log(extra["time"]/1000.) - self.time_intensity) / (self.attempts_count)
        self.time_intensity += TI_delta

        return prediction