import datetime
import json
import os
import numpy as np
from data.data import Data
from models.elo import EloModel
from data import utils
from models.model import AvgModel, AvgItemModel
from evaluator import Evaluator
import pandas as pd
from models.elo_corr import *
import pylab as plt


class Runner():
    def __init__(self, data, model):
        self.data = data
        self.model = model
        model.logger = self.pandas_logger
        self.logger_file = None #open("logs/{}.log".format(utils.hash(self.model, self.data)), "w")
        self.log = pd.Series(index=self.data.get_dataframe() .index)
        self.hash = utils.hash(self.model, self.data)

    def file_logger(self, log):
        self.logger_file.write("{}\n".format(log))

    def pandas_logger(self, answer, prediction):
        self.log[answer["id"]] = prediction

    def clean(self):
        os.remove("logs/{}.report".format(self.hash))
        os.remove("logs/{}.pd".format(self.hash))


    def run(self):
        start = datetime.datetime.now()
        print "Pre-processing data..."
        self.model.pre_process_data(self.data)
        pre_processing_time = datetime.datetime.now() - start
        print pre_processing_time

        start = datetime.datetime.now()
        print "Processing data..."
        self.model.process_data(self.data)
        processing_time = datetime.datetime.now() - start
        print processing_time

        report = {
            "model": str(self.model),
            "data": str(self.data),
            "processing time": str(processing_time),
            "pre-processing time": str(pre_processing_time),
            "data count": self.data.n,
        }

        with open("logs/{}.report".format(self.hash), "w") as f:
            json.dump(report, f)

        self.log.save("logs/{}.pd".format(self.hash))

        print "Written to {} report and log".format(self.hash)


def run_all_models(data, run=True):
    models = [EloModel(), AvgModel(), AvgItemModel(), EloCorrModel()]
    for model in models:
        print model
        if run:
            Runner(data, model).run()
        Evaluator(data, model).evaluate()


def elo_grid_search(data, run=True):
    alphas = np.arange(0.4, 2, 0.2)
    betas = np.arange(0.02, 0.2, 0.02)

    results = pd.DataFrame(columns=alphas, index=betas, dtype=float)
    for alpha in alphas:
        for beta in betas:

            model = EloModel(alpha=alpha, beta=beta)
            if run:
                Runner(data, model).run()
                report = Evaluator(data, model).evaluate()
            else:
                report = Evaluator(data, model).get_report()

            results[alpha][beta] = report["brier"]["reliability"]
            # results[alpha][beta] = report["rmse"]
    plt.figure()
    plt.title(data)
    plt.pcolor(results)
    plt.yticks(np.arange(0.5, len(results.index), 1), results.index)
    plt.ylabel("betas")
    plt.xticks(np.arange(0.5, len(results.columns), 1), results.columns)
    plt.xlabel("alphas")



def elo_corr_grid_search(data, run=True):

    prior_weights = np.arange(0, 2, 0.2)
    corr_place_weights = np.arange(0, 2, 0.2)

    results = pd.DataFrame(columns=prior_weights, index=corr_place_weights, dtype=float)
    for prior in prior_weights:
        for corr_place in corr_place_weights:

            model = EloCorrModel(prior_weight=prior, corr_place_weight=corr_place)
            if run:
                Runner(data, model).run()
                report = Evaluator(data, model).evaluate()
            else:
                report = Evaluator(data, model).get_report()

            results[prior][corr_place] = report["brier"]["reliability"]
    plt.figure()
    plt.title(data)
    plt.pcolor(results)
    plt.yticks(np.arange(0.5, len(results.index), 1), results.index)
    plt.ylabel("corr_place_weights")
    plt.xticks(np.arange(0.5, len(results.columns), 1), results.columns)
    plt.xlabel("prior_weights")

