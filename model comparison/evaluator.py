import json
import math
import os
import numpy as np
import pandas as pd
import pylab as plt
from data.data import Data
from models.elo import EloModel
from data import utils
from models.model import AvgModel, AvgItemModel
import runner


class Evaluator:
    def __init__(self, data, model):
        self.model = model
        self.data = data
        self.hash = utils.hash(model, data)

        if not os.path.isfile("logs/{}.report".format(self.hash)):
            print "Computing missing data {}; {}".format(data, model)
            runner.Runner(data, model).run()
            self.evaluate()

        print self.hash

    def evaluate(self, brier_bins=20):
        report = self.get_report()

        n = 0           # log count
        sse = 0         # sum of square error
        llsum = 0       # log-likely-hood sum
        brier_counts = np.zeros(brier_bins)          # count of answers in bins
        brier_correct = np.zeros(brier_bins)        # sum of correct answers in bins
        brier_prediction = np.zeros(brier_bins)     # sum of predictions in bins

        self.data.join_predictions(pd.load("logs/{}.pd".format(self.hash)))

        for log in self.data:
            n += 1
            sse += (log["prediction"] - log["correct"]) ** 2
            llsum += math.log(max(0.0001, log["prediction"] if log["correct"] else (1 - log["prediction"])))

            # brier
            bin = min(int(log["prediction"] * brier_bins), brier_bins - 1)
            brier_counts[bin] += 1
            brier_correct[bin] += log["correct"]
            brier_prediction[bin] += log["prediction"]

        answer_mean = sum(brier_correct) / n

        report["zextra"] = {"anser_mean": answer_mean}
        report["rmse"] = math.sqrt(sse / n)
        report["log-likely-hood"] = llsum

        # brier
        brier_prediction_means = brier_prediction / brier_counts
        brier_prediction_means[np.isnan(brier_prediction_means)] = \
            ((np.arange(brier_bins) + 0.5) / brier_bins)[np.isnan(brier_prediction_means)]
        brier_correct_means = brier_correct / brier_counts
        brier_correct_means[np.isnan(brier_correct_means)] = 0
        brier = {
            "reliability":  sum(brier_counts * (brier_correct_means - brier_prediction_means) ** 2) / n,
            "resolution":  sum(brier_counts * (brier_correct_means - answer_mean) ** 2) / n,
            "uncertainty": answer_mean * (1 - answer_mean),

        }
        report["brier"] = brier

        report["zextra"]["brier"] = {
            "bin_count": brier_bins,
            "bin_counts": list(brier_counts),
            "bin_prediction_means": list(brier_prediction_means),
            "bin_correct_means": list(brier_correct_means),
        }

        with open("logs/{}.report".format(self.hash), "w") as f:
            json.dump(report, f)

        return report

    def get_report(self):
        with open("logs/{}.report".format(self.hash)) as f:
            report = json.load(f)
        return report

    def save_report(self, report):
        with open("logs/{}.report".format(self.hash), "w") as f:
            json.dump(report, f)

    def __str__(self):
        return json.dumps(self.get_report(), sort_keys=True, indent=4, )

    def brier_graphs(self, show=True):
        report = self.get_report()

        plt.figure()
        plt.plot(report["zextra"]["brier"]["bin_prediction_means"], report["zextra"]["brier"]["bin_correct_means"])
        plt.plot((0, 1), (0, 1))

        bin_count = report["zextra"]["brier"]["bin_count"]
        counts = np.array(report["zextra"]["brier"]["bin_counts"])
        bins = (np.arange(bin_count) + 0.5) / bin_count
        plt.bar(bins, counts / max(counts), width=(0.5 / bin_count), alpha=0.5)
        plt.title(self.model)

        if show:
            plt.show()


def compare_models(data, models, dont=False, resolution=True):
    if dont:
        return
    plt.xlabel("RMSE")
    if resolution:
        plt.ylabel("Resolution")
    else:
        plt.ylabel("Brier score")
    for model in models:
        report = Evaluator(data, model).get_report()
        print model
        print "RMSE: {:.5}".format(report["rmse"])
        print "Brier resolution: {:.4}".format(report["brier"]["resolution"])
        print "Brier reliability: {:.3}".format(report["brier"]["reliability"])
        print "=" * 50

        x = report["rmse"]
        if resolution:
            y = report["brier"]["resolution"]
        else:
            y = report["brier"]["reliability"] - report["brier"]["resolution"] + report["brier"]["uncertainty"]
        plt.plot(x, y, "bo")
        plt.text(x, y, model, rotation=0, )


def compare_brier_curve(data, model1, model2):
    report1 = Evaluator(data, model1).get_report()
    report2 = Evaluator(data, model2).get_report()

    plt.figure()
    plt.plot(report1["zextra"]["brier"]["bin_prediction_means"], report1["zextra"]["brier"]["bin_correct_means"], "g", label="M1")
    plt.plot(report2["zextra"]["brier"]["bin_prediction_means"], report2["zextra"]["brier"]["bin_correct_means"], "b", label="M2")
    plt.plot((0, 1), (0, 1), "r")

    bin_count = report1["zextra"]["brier"]["bin_count"]
    counts1 = np.array(report1["zextra"]["brier"]["bin_counts"])
    counts2 = np.array(report2["zextra"]["brier"]["bin_counts"])
    bins = (np.arange(bin_count) + 0.5) / bin_count
    plt.bar(bins, counts1 / max(max(counts2),max(counts1)), width=(0.5 / bin_count), alpha=0.2, color="g")
    plt.bar(bins, counts2 / max(max(counts2),max(counts1)), width=(0.5 / bin_count), alpha=0.2, color="b")
    plt.bar(bins, (counts1 - counts2) / max(max(counts2),max(counts1)), width=(0.5 / bin_count), alpha=0.8, color="r")

    plt.title("{}\n{}".format(model1, model2))

    plt.legend(loc=2)

def group_rmse(data, models, groups, dont=False):
    if dont:
        return
    from hashlib import sha1
    plt.figure()
    plt.ylabel("RMSE")
    colors = ["blue", "red", "green", "black", "cyan", "yellow", "purple"]

    for i, model in enumerate(models):
        groups_hash = "group_{}".format(sha1(";".join(groups.keys())).hexdigest()[:10])
        report = Evaluator(data, model).get_report()
        if groups_hash not in report:
            group_map = {}
            groups_names = []
            for group, items in groups.items():
                groups_names.append(group)
                for item in items:
                    group_map[item] = group

            n = pd.Series(index=groups_names).fillna(0)           # log count
            sse = pd.Series(index=groups_names).fillna(0)         # sum of square error

            data.join_predictions(pd.load("logs/{}.pd".format(utils.hash(model, data))))

            for log in data:
                group = group_map[log["item"]]
                n[group] += 1
                sse[group] += (log["prediction"] - log["correct"]) ** 2

            report[groups_hash] = (sse / n).apply(math.sqrt).to_dict()
            Evaluator(data, model).save_report(report)

        s = pd.Series()
        for g, v in report[groups_hash].items():
            s[g] = v

        s = s[s.notnull()]

        plt.bar(np.arange(len(s))+i*0.8/len(models), s.values, color=colors[i], width=0.8/len(models), label=str(model)[:50])
        plt.xticks(np.arange(len(s))+0.5, s.index, rotation=-90)

    plt.legend(loc=4)