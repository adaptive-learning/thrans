from model import Model, sigmoid


class EloTreeModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, clusters=None, decay_function=None, local_update_boost=1):
        if not clusters: clusters = {}
        Model.__init__(self)

        self.alpha = alpha
        self.beta = beta
        self.decay_function = decay_function if decay_function is not None else lambda x: alpha / (1 + beta * x)
        self.local_update_boost = local_update_boost

        self.global_skill = {}
        self.maps_skills = {}
        self.difficulty = {}
        self.student_attempts = {}
        self.map_student_attempts = {}
        self.place_attempts = {}
        self.clusters = clusters

        self.maps_map = {}
        for map, places in self.clusters.items():
            self.maps_skills[map] = {}
            self.map_student_attempts[map] = {}
            for place in places:
                self.maps_map[place] = map

    def __str__(self):
        return "Elo tree v2; decay - alpha: {}, beta: {}, local_update_boost: {}, clusters: {}".format(self.alpha, self.beta, self.local_update_boost, ";".join(self.clusters.keys()))

    def initialize_if_needed(self, student, item):
        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.student_attempts[student] = 0

        map = self.maps_map[item]
        if not student in self.maps_skills[map]:
            self.maps_skills[map][student] = 0
            self.map_student_attempts[map][student] = 0
        if not item in self.difficulty:
            self.difficulty[item] = 0
            self.place_attempts[item] = 0

    def process(self, student, item, correct, extra=None):
        self.initialize_if_needed(student, item)
        map = self.maps_map[item]
        random_factor = 0 if extra is None or extra["choices"] == 0 else 1. / extra["choices"]

        skill = self.global_skill[student] + self.maps_skills[map][student]
        prediction = sigmoid(skill - self.difficulty[item], random_factor)
        dif = (correct - prediction)
        update = self.decay_function(self.student_attempts[student]) * dif
        self.global_skill[student] += update

        skill = self.global_skill[student] + self.maps_skills[map][student]
        prediction2 = sigmoid(skill - self.difficulty[item], random_factor)
        dif = (correct - prediction2)
        update = self.decay_function(self.map_student_attempts[map][student]) * dif * self.local_update_boost
        self.maps_skills[map][student] += update


        self.difficulty[item] -= self.decay_function(self.place_attempts[item]) * dif
        self.map_student_attempts[map][student] += 1
        self.student_attempts[student] += 1
        self.place_attempts[item] += 1

        return prediction