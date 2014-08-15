import random

# pandas read_csv vypada trochu rychlejsi, ale zas ne o moc, neresim...

def get_csv_head_map(line):
    m = {}
    p = line.rstrip().split(',')
    for i in range(len(p)): m[p[i]] = i
    return m


def split(datafile, p=0.5):
    with open(datafile) as f:
        f1 = open(datafile.replace(".csv", "-{0}-1.csv".format(p)), "w")
        f2 = open(datafile.replace(".csv", "-{0}-2.csv".format(1-p)), "w")
        head = f.readline()
        f1.write(head)
        f2.write(head)
        for line in f:
            student = line.split(',')[0]
            random.seed(student)
            if random.random() < p:
                f1.write(line)
            else:
                f2.write(line)


class Data:
    def __init__(self, datafile="processed-data/data.csv", dataframe=None, places=None):
        self.n = 0
        self.places = set()
        self.users = set()
        self.datafile = datafile
        if datafile:
            self.read(datafile, places)
        if dataframe:
            self.init_columns(dataframe.columns)

    def __str__(self):
        return self.datafile+".{0}".format(sum(self.places))

    def init_columns(self, col):
        self.columns = col
        for c in self.columns: self.__dict__[c] = []
            
    def read(self, datafile, places):
        f = open(datafile)
        self.init_columns(get_csv_head_map(f.readline()))
        for line in f:
            p = line.split(',')
            if places is None or int(p[self.columns["place"]]) in places:
                for c in self.columns:
                    val = int(p[self.columns[c]])
                    if c == 'rtime':
                        val = max(200, min(30000, val))
                    self.__dict__[c].append(val)
                self.n += 1
                self.places.add(int(p[self.columns["place"]]))
                self.users.add(int(p[self.columns["student"]]))

        f.close()

if False:
    split("data-first.csv", p=0.8)