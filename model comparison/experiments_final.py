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

data_off = Data("data/geographyOF-first.pd", train=0.3, force_train=42)
data_off_hetero = Data("data/geographyOF-first-hetero2.pd", train=0.3, force_train=42)
data_off_homo = Data("data/geographyOF-first-homo.pd", train=0.3, force_train=42)
data_off_europe = Data("data/geographyOF-first-europe.pd", train=0.3, force_train=42)
data_all = Data("data/geography-first-all-2.pd", train=0.3)
data_train = Data("data/geography-first-all-2.pd", train=0.3, only_train=True)
data_train2 = Data("data/geography-first-all-2-train.pd", train=0.5, force_train=True)


# compare_brier_curve(data_all, EloModel(beta=0.06), EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),)
# print len(data_all.get_items()), len(data_all.get_students()), len(data_all.get_dataframe_all())

# print len(data_off.get_dataframe_all())
# print len(data_off.get_dataframe_all()["student"].unique())

compare_models(data_off, [
    # AvgModel(),
    # AvgItemModel(),
    # EloModel(),
    # EloModel(beta=0.06),
    # EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    # EloCorrModel(corr_place_weight=0.4, prior_weight=1),
    # EloCorrModel(corr_place_weight=0.4, prior_weight=1, min_corr=200),
    EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),
    # EloTreeModel(clusters=get_maps("data/", old=True), local_update_boost=0.5),
    # EloTreeModel(clusters=get_maps("data/", just_types=True), local_update_boost=0.5),
    # EloTreeModel(clusters=get_continents_country_maps(folder="data/"), local_update_boost=0.5),
], dont=1, auc=False, resolution=True, evaluate=0, diff_to=0.4142)

compare_models(data_off_hetero, [
    # AvgModel(),
    # AvgItemModel(),
    # EloModel(),
    EloModel(beta=0.06),
    # EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    # EloCorrModel(corr_place_weight=0.4, prior_weight=1),
    # EloCorrModel(corr_place_weight=0.4, prior_weight=1, min_corr=200),
    # EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200, corrfile="data/2e23b47131 min_corr: 200.corr.pd"),
    EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),
    # EloTreeModel(clusters=get_maps("data/", old=True), local_update_boost=0.5),
    # EloTreeModel(clusters=get_maps("data/", just_types=True), local_update_boost=0.5),
    # EloTreeModel(clusters=get_continents_country_maps(folder="data/"), local_update_boost=0.5),
], dont=0, auc=False, resolution=True, evaluate=0, diff_to=0.41814, rerun=0)


compare_models(data_off_homo, [
    # AvgModel(),
    # AvgItemModel(),
    # EloModel(),
    EloModel(beta=0.06),
    # EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    # EloCorrModel(corr_place_weight=0.4, prior_weight=1),
    # EloCorrModel(corr_place_weight=0.4, prior_weight=1, min_corr=200),
    # EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200, corrfile="data/2e23b47131 min_corr: 200.corr.pd"),
    EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),
    # EloTreeModel(clusters=get_maps("data/", old=True), local_update_boost=0.5),
    # EloTreeModel(clusters=get_maps("data/", just_types=True), local_update_boost=0.5),
    # EloTreeModel(clusters=get_continents_country_maps(folder="data/"), local_update_boost=0.5),
], dont=0, auc=False, resolution=True, evaluate=0, diff_to=0.41697, rerun=0)

# elo_grid_search(data_off_europe)
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
    d1 = pd.load("logs/{}.pd".format(utils.hash(model1, data_off)))
    d2 = pd.load("logs/{}.pd".format(utils.hash(model2, data_off)))
    ch = np.random.choice(d1.index, 3000)
    # plt.plot(d1.ix[ch], d2.ix[ch], ".")
    d = abs(d1 - d2)
    # data_all.join_predictions(d)
    # d = data_all.data[data_all.data["prediction"] > 0.1]
    # print len(d), len(d["student"].unique()), len(data_all.data["student"].unique()), len(data_all.data)

    print 1. * (d<0.1).sum() / len(d)


# d = data_all.get_train_dataframe()
# students = d["student"].unique()
# selected_students = random.sample(students, int(len(students) * 0.5))
# d.to_pickle("data/geography-first-all-2-train.pd")

# corr_stats(data_train2, min_periods=200, test_dataset=True)
# hits = pd.read_pickle("tmp.hits")
# hits.unstack().hist(bins=30)

plt.show()

def ban_homogeneous_students():
    students = data_off.get_dataframe_all().groupby("student").last()

    all_students = set()
    for i, r in sorted(students.groupby("ip_country").apply(len).iteritems()):
        print i, r
        all_students |= set(students[students["ip_country"] == i].index)

    banned_students = set()
    for nat in ["CZ", "SK"]:
        s = students[students["ip_country"] == nat].index
        banned_students |= set(random.sample(list(s), len(s) - 1000))

    df = data_off.get_dataframe_all()
    print len(df)
    df = df[df["student"].isin(all_students - banned_students)]
    print len(df)
    df.to_pickle("data/geographyOF-first-hetero2.pd")

def select_homogeneous_students():
    students = data_off.get_dataframe_all().groupby("student").last()

    s = students[students["ip_country"] == "CZ"].index
    s = random.sample(list(s), 6000)

    df = data_off.get_dataframe_all()
    print len(df)
    df = df[df["student"].isin(s)]
    print len(df)
    df.to_pickle("data/geographyOF-first-homo.pd")


# ban_homogeneous_students()
# select_homogeneous_students()