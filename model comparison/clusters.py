import sys
sys.path.append("../Algoritmy")
import spectralclustering as sc
from evaluator import *
from data.data import *
import numpy as np
from models.elo_corr import *
from models.elo_tree import *
import pylab as plt
from sklearn import metrics, linear_model

colors = ["blue", "red", "green", "cyan", "purple", "black", "orange", "magenta", "gray", "yellow"]

places_all = places.from_csv("data/raw data/geography.place.csv", "data/raw data/geography.placerelation.csv","data/raw data/geography.placerelation_related_places.csv")
places_all.set_index(places_all["id"], inplace=True)
data_all = Data("data/geography-first-all-2.pd", train=0.3)

def random_concepts(all, concepts=3):
    np.random.seed(42)
    split = np.random.random_integers(0, concepts-1, size=len(all))
    c = {}
    all = np.array(all)
    for i in range(concepts):
        c["random"+str(i)] = all[split==i]

    return c

def compute_clusters(data, clusters=3, run=False, plot=False):
    filename = "data/{}.clust.json".format(sha1(str(data) + str(clusters)).hexdigest()[:10])
    try:
        if not run:
            return json.load(open(filename))
    except:
        pass

    filename2 = "data/{}.corr.pd".format(sha1(str(data)).hexdigest()[:10])
    try:
        corr = pd.read_pickle(filename2)
        print "Loaded corr matrix"
    except:
        print "Computing corr matrix"
        corr, _, nans = compute_correlations(data, min_periods=200)
        corr.to_pickle(filename2)

    # print corr.shape,
    filt = (corr > 0).sum() > 2
    corr = corr.ix[filt, filt]
    # print corr.shape
    labels, SC = sc.clusterSpearmanSC(corr, clusterNumber=clusters, KMiter=50, kcut=corr.shape[0]/2, SCtype=2, plot = True, mutual = False)


    clusters = defaultdict(list)
    for i, c in enumerate(labels):
        clusters[str(c)].append(corr.index[i])

    if len(colors) > len(clusters.keys()) and plot:
        plt.figure()
        for n, p in enumerate(corr.columns):
            place = places_all.loc[places_all["id"]==p].squeeze()
            if type(place["name_en"]) == str:
                plt.plot(SC.eig_vect[n,1],SC.eig_vect[n,2], "o", color=colors[labels[n]])
                plt.text(SC.eig_vect[n,1],SC.eig_vect[n,2], place["name_en"].decode("utf8"))

    json.dump(clusters, open(filename, "w"))
    return clusters


def correct_cluster(data, clusters, regularization=2, run=False, print_diffs=False):
    filename = "data/{}.clust.corr.json".format(sha1(str(data) + str(clusters)+str(regularization)).hexdigest()[:10])
    try:
        if not run:
            print "Loaded corrected clusters", filename
            return json.load(open(filename))
    except:
        pass

    filename2 = "data/{}.corr.pd".format(sha1(str(data)).hexdigest()[:10])
    try:
        corr = pd.read_pickle(filename2)
        print "Loaded corr matrix"
    except:
        print "Computing corr matrix"
        corr, _, nans = compute_correlations(data, min_periods=200)
        corr.to_pickle(filename2)

    clusters["other"] = []
    names = clusters.keys()

    labels = []
    for id in corr.index:
        done = False
        for i, c in enumerate(names):
            places = clusters[c]
            if id in places:
                labels.append(i)
                done = True
        if not done:
            labels.append(names.index("other"))
    labels = np.array(labels)


    clf = linear_model.LogisticRegression(C=regularization)
    clf.fit(corr, labels)
    corrected_labels = clf.predict(corr)
    corrected = defaultdict(lambda: [])


    for id, label in zip(corr.index, corrected_labels):
        corrected[names[label] + " corr " + str(regularization)].append(id)

    print (corrected_labels != labels).sum()

    if print_diffs:
        for i, place in enumerate(corr.index):
            if labels[i] != corrected_labels[i]:
                print "{} - {} -> {}".format(places_all.loc[place, "name"], names[labels[i]], names[corrected_labels[i]]), place

    json.dump(corrected, open(filename, "w"))
    return corrected


def cluster_optim(data, clusters=None, values=range(2, 11), local_update_boost=0.5):
    plt.figure()
    plt.title(str(data))
    elo = Evaluator(data, EloModel()).get_report()["rmse"]
    plt.plot([values[0], values[-1]], [elo, elo], "b-", label="elo")

    for c, color in zip(clusters, colors[2:]):
        exp_tree = Evaluator(data, EloTreeModel(clusters=c, local_update_boost=local_update_boost),).get_report()["rmse"]
        plt.plot([values[0], values[-1]], [exp_tree, exp_tree], "-", color=color, label="expert clusters ({})".format(len(c)))

        exp_tree_corr = Evaluator(data, EloTreeModel(clusters=correct_cluster(data, c), local_update_boost=local_update_boost),).get_report()["rmse"]
        plt.plot([values[0], values[-1]], [exp_tree_corr, exp_tree_corr], "--", color=color, label="expert corrected clusters ({})".format(len(c)))

    results = []
    for v in values:
        r = Evaluator(data, EloTreeModel(clusters=compute_clusters(data_all, v), local_update_boost=local_update_boost),).get_report()["rmse"]
        results.append(r)

    plt.plot(values, results, "r-", label="spectral clusters")
    plt.xlabel("concepts")
    plt.ylabel("rmse")
    plt.legend(loc=4)