# coding=utf-8
from runner import *
from data.utils import *
from evaluator import *
from data.data import *
from models.elo_corr import *
from models.elo_tree import *
from models.elo_clust import *
from models.elo_time import *
import pylab as plt
import scipy.stats as sc

data_all = Data("data/geography-first-all-2.pd", train=0.3)
data_train = Data("data/geography-first-all-2.pd", train=0.3, only_train=True)
data_train2 = Data("data/geography-first-all-2-train.pd", train=0.5, force_train=True)


# compare_brier_curve(data_all, EloModel(beta=0.06), EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),)
# print len(data_all.get_items()), len(data_all.get_students()), len(data_all.get_dataframe_all())

compare_models(data_all, [
    # AvgModel(),
    # AvgItemModel(),
    EloModel(),
    EloModel(beta=0.06),
    # EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    # EloCorrModel(corr_place_weight=0.4, prior_weight=1),
    # EloCorrModel(corr_place_weight=0.4, prior_weight=1, min_corr=200),
    EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),
    # EloTreeModel(clusters=get_maps("data/", old=True), local_update_boost=0.5),
    # EloTreeModel(clusters=get_maps("data/", just_types=True), local_update_boost=0.5),
    # EloTreeModel(clusters=get_continents_country_maps(folder="data/"), local_update_boost=0.5),
    ], dont=1, auc=False, resolution=True, evaluate=0)

# elo_grid_search(data_train)
# elo_corr_grid_search(data_train2)

if False:
    clusters = get_maps("data/")
    model = EloTreeModel(clusters=clusters, local_update_boost=0.5)
    Runner(data_all, model).run()
    global_skills = pd.Series(model.global_skill, index=model.global_skill.keys())

    skills = pd.DataFrame(data=global_skills, columns=["global"])
    for map in clusters.keys():
        local_skills = pd.Series(model.maps_skills[map], index=model.maps_skills[map].keys())
        skills[map] = local_skills

    skills = skills[(~skills.isnull()).sum(axis=1) > 1]

    values = []
    for map in clusters.keys():
        if (~skills[map].isnull()).sum() == 0:
            continue

        # plt.figure()
        # tmp = skills[~skills[map].isnull()]
        # plt.title(skills.corr().ix["global", map])
        values.append(skills.corr().ix["global", map])
        # plt.plot(tmp["global"], tmp[map], ".")
        # plt.ylabel(map)
        # plt.xlabel("global")
        # plt.savefig("results/tree-elo skill-corr {} filtered-6.png".format(map))

    print values
    plt.hist(values, bins=50)


if False:
    model1 = EloModel(beta=0.06)
    model2 = EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5)
    runs = 30
    results1 = []
    results2 = []

    for seed in range(1,runs + 1):
        data = Data("data/geography-first-all-2.pd", train=0.3, force_train=seed)
        results1.append(Evaluator(data, model1).get_report()["rmse"])
        results2.append(Evaluator(data, model2).get_report()["rmse"])

    print sc.ttest_ind(results1, results2)
    plt.plot(results1, results2, ".")


if False:
    params = np.arange(0.2, 1, 0.05)
    results = []
    for param in params:
        model = EloTreeModel(clusters=get_maps("data/"), local_update_boost=param)
        results.append(Evaluator(data_train, model).get_report()["rmse"])

    plt.figure()
    plt.plot(params, results, '-')

if False:
    model1 = EloModel(beta=0.06)
    # model2 = EloCorrModel(corr_place_weight=0.4, prior_weight=1)
    model2 = EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5)
    # model2 = EloModel()
    # model2 = EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5)
    d1 = pd.load("logs/{}.pd".format(utils.hash(model1, data_all)))
    d2 = pd.load("logs/{}.pd".format(utils.hash(model2, data_all)))
    ch = np.random.choice(d1.index, 3000)
    # plt.plot(d1.ix[ch], d2.ix[ch], ".")
    d = abs(d1 - d2)
    # data_all.join_predictions(d)
    # d = data_all.data[data_all.data["prediction"] > 0.1]
    # print len(d), len(d["student"].unique()), len(data_all.data["student"].unique()), len(data_all.data)

    print 1. * (d<0.01).sum() / len(d)

if True:
    pass

# d = data_all.get_train_dataframe()
# students = d["student"].unique()
# selected_students = random.sample(students, int(len(students) * 0.5))
# d.to_pickle("data/geography-first-all-2-train.pd")

# corr_stats(data_train2, min_periods=200, test_dataset=True)
# hits = pd.read_pickle("tmp.hits")
# hits.unstack().hist(bins=30)

plt.show()