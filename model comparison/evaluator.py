import json
import math
import numpy as np
import pandas as pd
import pylab as plt
import scipy
from sklearn import metrics
from data.data import Data
from data.utils import *
from models.elo import EloModel
from data import utils
from models.model import AvgModel, AvgItemModel
import runner
from hashlib import sha1


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

    def delete(self):
        os.remove("logs/{}.pd".format(self.hash))
        os.remove("logs/{}.report".format(self.hash))

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
        report["AUC"] = metrics.roc_auc_score(self.data.get_dataframe()["correct"], self.data.get_dataframe()   ["prediction"])

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

    def   get_report(self):
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


def compare_models(data, models, dont=False, resolution=True, auc=False, evaluate=False, diff_to=None):
    if dont:
        return
    plt.xlabel("RMSE")
    if auc:
        plt.ylabel("AUC")
    elif resolution:
        plt.ylabel("Resolution")
    else:
        plt.ylabel("Brier score")
    for model in models:
        if evaluate:
            Evaluator(data, model).evaluate()
        report = Evaluator(data, model).get_report()
        print model
        print "RMSE: {:.5}".format(report["rmse"])
        if diff_to is not None:
            print "RMSE diff: {:.5f}".format(diff_to - report["rmse"])
        print "LL: {:.6}".format(report["log-likely-hood"])
        print "AUC: {:.4}".format(report["AUC"])
        print "Brier resolution: {:.4}".format(report["brier"]["resolution"])
        print "Brier reliability: {:.3}".format(report["brier"]["reliability"])
        print "=" * 50

        x = report["rmse"]
        if auc:
            y = report["AUC"]
        elif resolution:
            y = report["brier"]["resolution"]
        else:
            y = report["brier"]["reliability"] - report["brier"]["resolution"] + report["brier"]["uncertainty"]
        plt.plot(x, y, "bo")
        plt.text(x, y, model, rotation=0, )


def compare_brier_curve(data, model1, model2):
    # Evaluator(data, model2).evaluate()
    report1 = Evaluator(data, model1).get_report()
    report2 = Evaluator(data, model2).get_report()

    fig, ax1 = plt.subplots()
    ax1.plot([]+report1["zextra"]["brier"]["bin_prediction_means"], []+report1["zextra"]["brier"]["bin_correct_means"], "g", label="M1")
    ax1.plot([]+report2["zextra"]["brier"]["bin_prediction_means"], []+report2["zextra"]["brier"]["bin_correct_means"], "r", label="M2")
    ax1.plot((0, 1), (0, 1), "k--")

    ax2 = ax1.twinx()

    bin_count = report1["zextra"]["brier"]["bin_count"]
    counts1 = np.array(report1["zextra"]["brier"]["bin_counts"])
    counts2 = np.array(report2["zextra"]["brier"]["bin_counts"])
    bins = (np.arange(bin_count) + 0.5) / bin_count
    ax2.bar(bins, counts1, width=(0.45 / bin_count), alpha=0.2, color="g")
    ax2.bar(bins-0.023, counts2, width=(0.45 / bin_count), alpha=0.2, color="r")
    # plt.bar(bins, (counts1 - counts2) / max(max(counts2),max(counts1)), width=(0.5 / bin_count), alpha=0.8, color="r")

    # plt.title("{}\n{}".format(model1, model2))
    plt.xticks(list(bins-0.025) + [1.])

    plt.legend(loc=2)


def get_group_map(groups):
    group_map = {}
    groups_names = []
    for group, items in groups.items():
        groups_names.append(group)
        for item in items:
            group_map[item] = group

    return group_map, groups_names


def group_calibration(data, models, groups, dont=False):
    if dont:
        return
    plt.figure()
    colors = ["blue", "red", "green", "cyan", "yellow", "purple", "black",]

    group_map, groups_names = get_group_map(groups)

    for i, model in enumerate(models):
        group_map, groups_names = get_group_map(groups)

        real = pd.Series(index=groups_names).fillna(0)         # sum success
        real_std = pd.Series(index=groups_names).fillna(0)         # sum success
        predicted = pd.Series(index=groups_names).fillna(0)         # sum predicted
        predicted_std = pd.Series(index=groups_names).fillna(0)         # sum predicted

        data.join_predictions(pd.read_pickle("logs/{}.pd".format(utils.hash(model, data))))
        df = data.get_dataframe()
        for name, group in groups.items():
            real[name] = df[df["item"].isin(group)]["correct"].mean()
            # real_std[name] = df[df["item"].isin(group)]["correct"].std()
            predicted[name] = df[df["item"].isin(group)]["prediction"].mean()
            # predicted_std[name] = df[df["item"].isin(group)]["prediction"].std()

        real = real[real.notnull()]
        predicted = predicted[predicted.notnull()]

        plt.bar(np.arange(len(predicted))+((i+1)*0.8)/(len(models)+1), (real - predicted).values, color=colors[i+1], width=0.8/(len(models) + 1), label=str(model)[:50])
        plt.title("observed - predicted")

    plt.xticks(np.arange(len(predicted))+0.5, predicted.index, rotation=-90)
    # plt.bar(np.arange(len(real)), real.values, color=colors[0], width=0.8/(len(models) + 1), label="observed", hatch="///")

    plt.legend(loc=4)

def group_rmse(data, models, groups, dont=False):
    if dont:
        return
    plt.figure()
    plt.ylabel("RMSE")
    colors = ["blue", "red", "green", "black", "cyan", "yellow", "purple"]

    for i, model in enumerate(models):
        groups_hash = "group_{}".format(sha1(";".join(groups.keys())).hexdigest()[:10])
        report = Evaluator(data, model).get_report()
        if groups_hash not in report:
            group_map, groups_names = get_group_map(groups)

            n = pd.Series(index=groups_names).fillna(0)           # log count
            sse = pd.Series(index=groups_names).fillna(0)         # sum of square error

            data.join_predictions(pd.load("logs/{}.pd".format(utils.hash(model, data))))

            for log in data:
                try:
                    group = group_map[log["item"]]
                    n[group] += 1
                    sse[group] += (log["prediction"] - log["correct"]) ** 2
                except:
                    print "skipping", log["item"]

            report[groups_hash] = (sse / n).apply(math.sqrt).to_dict()
            Evaluator(data, model).save_report(report)

        s = pd.Series()
        for g, v in report[groups_hash].items():
            s[g] = v

        s = s[s.notnull()]

        plt.bar(np.arange(len(s))+i*0.8/len(models), s.values, color=colors[i], width=0.8/len(models), label=str(model)[:50])
        plt.xticks(np.arange(len(s))+0.5, s.index, rotation=-90)

    plt.legend(loc=4)


def options_rmse(data, models, nothing=None, dont=False):
    if dont:
        return
    plt.figure()
    plt.ylabel("RMSE")
    colors = ["blue", "red", "green", "black", "cyan", "yellow", "purple"]

    for i, model in enumerate(models):
        report = Evaluator(data, model).get_report()
        option_field_name = "options-count-rmse"
        if option_field_name not in report:

            max_choices = data.get_dataframe()["choices"].max()

            n = pd.Series(index=range(max_choices+1)).fillna(0)           # log count
            sse = pd.Series(index=range(max_choices+1)).fillna(0)         # sum of square error

            data.join_predictions(pd.load("logs/{}.pd".format(utils.hash(model, data))))

            for log in data:
                group = log["choices"]
                n[group] += 1
                sse[group] += (log["prediction"] - log["correct"]) ** 2

            report[option_field_name] = (sse / n).apply(math.sqrt).to_dict()
            Evaluator(data, model).save_report(report)

        s = pd.Series()
        for g, v in report[option_field_name].items():
            s[str(g)] = v

        s = s[s.notnull()]
        s["open"] = s["0"]
        s = s.sort_index()
        del s["0"]

        plt.bar(np.arange(len(s))+i*0.8/len(models), s.values, color=colors[i], width=0.8/len(models), label=str(model)[:50])
        plt.xticks(np.arange(len(s))+0.5, s.index, rotation=-90)

    plt.legend(loc=4)


def options_calibration(data, models, nothing=None, dont=False):
    if dont:
        return
    plt.figure()
    plt.ylabel("RMSE")
    colors = ["blue", "red", "green", "black", "cyan", "yellow", "purple"]

    for i, model in enumerate(models):
        report = Evaluator(data, model).get_report()
        option_field_name = "options-count-calibration"
        if option_field_name+"pred" not in report:

            max_choices = data.get_dataframe()["choices"].max()

            n = pd.Series(index=range(max_choices+1)).fillna(0)           # log count
            prediction = pd.Series(index=range(max_choices+1)).fillna(0)
            observed = pd.Series(index=range(max_choices+1)).fillna(0)

            data.join_predictions(pd.load("logs/{}.pd".format(utils.hash(model, data))))

            for log in data:
                group = log["choices"]
                n[group] += 1
                prediction[group] += log["prediction"]
                observed[group] += log["correct"]

            report[option_field_name+"pred"] = (prediction/ n).apply(math.sqrt).to_dict()
            report[option_field_name+"observ"] = (observed/ n).apply(math.sqrt).to_dict()
            Evaluator(data, model).save_report(report)

        s = pd.Series()
        t = pd.Series()
        for g, v in report[option_field_name+"pred"].items():
            s[str(g)] = v

        for g, v in report[option_field_name+"observ"].items():
            t[str(g)] = v

        s = s - t

        s = s[s.notnull()]
        s["99"] = s["0"]
        s = s.sort_index()
        del s["0"]

        plt.bar(np.arange(len(s))+i*0.8/len(models), s.values, color=colors[i], width=0.8/len(models), label=str(model)[:50])
        plt.xticks(np.arange(len(s))+0.5, s.index, rotation=-90)

    plt.legend(loc=4)

def corr_stats(data, min_periods=100, test_dataset=True):
    # corr = pd.read_pickle("data/{}.corr.pd".format(sha1(str(data)).hexdigest()[:10]))
    corr, hits, nans = compute_correlations(data, guess_decay=True, min_periods=min_periods, test=test_dataset, hits=True)
    corr.to_pickle("tmp.corr")
    hits.to_pickle("tmp.hits")

    map = get_id_place_map("data/")
    corr.unstack().hist(bins=30)
    hits.unstack().hist(bins=30)
    # plt.plot(corr.unstack(), hits.unstack(), ".")

    name = "Dataset: {}, min common students: {}, only train set: {}".format(data, min_periods, test_dataset).replace("/", "-")
    for id, place in corr.iteritems():
        corr.ix[id, "name"] = map[id]



    with open("results/correlations/best_ten "+name+".txt", "w") as f:
        f.write("NaNs in correlation matrix: " + str(nans)+"\n")
        corr.sort("name", inplace=True)
        for id, place in corr.iterrows():
            if id != "name":
                place = place.order(ascending=False)
                if id != place.index[0]:
                    "Something went wrong"
                print map[id],  [(map[id2], round(place[id2],3), hits.ix[id, id2]) for id2 in place.index[1:12]]
                f.write(map[id]+" - "+str([(map[id2], round(place[id2],3), hits.ix[id, id2]) for id2 in place.index[1:12]])+"\n")
