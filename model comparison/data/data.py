import json


class Data():
    def __init__(self, filename, test=False):
        self.file = filename
        self.test = test
        self.n = -1         # not counted yet

    def __str__(self):
        if self.test:
            return "Test: " + self.file
        return self.file

    def __iter__(self):
        self.n = 0
        with open(self.file) as f:
            for line in f.readlines():
                self.n += 1
                yield json.loads(line)
                if self.n % 10000 == 0:
                    print ".",
                if self.test and self.n >= 10000:
                    break
