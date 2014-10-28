import datetime
import json
from data.data import Data
from models.elo import EloModel
from data import utils
from models.model import AvgModel


class Runner():
    def __init__(self, data, model):
        self.data = data
        self.model = model
        model.logger = self.file_logger
        self.logger_file = open("logs/{}.log".format(utils.hash(self.model, self.data)), "w")

    def file_logger(self, log):
        self.logger_file.write("{}\n".format(log))

    def run(self):
        start = datetime.datetime.now()
        print "Pre-processing data..."
        self.model.pre_process_data(self.data)
        print
        pre_processing_time = datetime.datetime.now() - start
        print pre_processing_time

        start = datetime.datetime.now()
        print "Processing data..."
        self.model.process_data(self.data)
        print
        processing_time = datetime.datetime.now() - start
        print processing_time

        report = {
            "model": str(self.model),
            "data": str(self.data),
            "processing time": str(processing_time),
            "pre-processing time": str(pre_processing_time),
            "data count": self.data.n,
        }

        with open("logs/{}.report".format(utils.hash(self.model, self.data)), "w") as f:
            json.dump(report, f)

        print "Written to {} report and log".format(utils.hash(self.model, self.data))

# Runner(Data("data/geography-first-all.json", test=False), EloModel()).run()
# Runner(Data("data/geography-first-all.json", test=False), AvgModel()).run()