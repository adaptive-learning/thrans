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
data_states = Data("data/geography-first-states-filtered-2.pd", train=0.3)
data_states_old = Data("data/geography-first-states-filtered.pd", train=0.3)
data_europe = Data("data/geography-first-europe-filtered-2.pd", train=0.3)
data_europe_old = Data("data/geography-first-europe-filtered.pd", train=0.3)
data_cz_cities = Data("data/geography-first-cz_city-filtered-2.pd", train=0.3)
data_cz_cities_old = Data("data/geography-first-cz_city-filtered.pd", train=0.3)
maps_continents_country = get_continents_country_maps("data/")

model = EloModel()
Runner(data_all, model).run()
pd.DataFrame(model.difficulty.items()).to_csv("difficulties.csv")
# Runner(Data("data/geography-first-all.pd", test=False), AvgModel()).run()
# Runner(Data("data/geography-first-all.pd", test=False), AvgItemModel()).run()

# model = EloCorrModel()
# Runner(data_states, model).run()
# experiment = Evaluator(data_states, model)
# experiment.evaluate()


# run_all_models(data, run=True)

# experiment = Evaluator(data_states, EloClusterModel(clusters=maps_continents_country))
# experiment = Evaluator(data_cz_cities, EloModel(alpha=0.2, beta=0.08))
# experiment = Evaluator(data_cz_cities, EloCorrModel(alpha=1, beta=0.08, corr_place_weight=0.4, prior_weight=0.6))

# experiment.evaluate()
# experiment.brier_graphs()
# compare_brier_curve(data_states, EloModel(), EloModel(alpha=1.2))
# compare_brier_curve(data_states, EloModel(), EloTimeModel())
# compare_brier_curve(data_states, EloModel(), EloClusterModel(clusters=maps_continents_country))
# compare_brier_curve(data_states, EloModel(), EloCorrModel(corr_place_weight=0.8, prior_weight=0.8))
# compare_brier_curve(data_states, EloModel(), EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4))
# compare_brier_curve(data_states, EloCorrModel(), EloCorrModel(corr_place_weight=0.8, prior_weight=0.8))
# print experiment

# compare_models(data_europe, [AvgModel(), AvgItemModel(), EloModel(), EloCorrModel(), EloCorrModel(corr_place_weight=1.8)])
# compare_models(data_states, [AvgModel(), AvgItemModel(), EloModel(), EloCorrModel(corr_place_weight=0.6, prior_weight=0.6), EloClusterModel(clusters=maps_continents_country)])

compare_models(data_all, [
    AvgModel(),
    AvgItemModel(),
    EloTimeModel(),
    EloModel(alpha=1.2),
    EloClusterModel(clusters=get_maps("data/")),
    EloClusterModel(clusters=get_maps("data/", just_types=True)),
    EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),
    EloTreeModel(clusters=get_maps("data/", just_types=True), local_update_boost=0.5),
], dont=True, resolution=True)

tmp_maps = get_maps("data/", just_types=True)
group_rmse(data_all, [
    # AvgModel(),
    # AvgItemModel(),
    # EloTimeModel(),
    EloModel(alpha=1.2),
    # EloClusterModel(clusters=tmp_maps),
    EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    EloTreeModel(clusters=tmp_maps, local_update_boost=0.5),
    ], tmp_maps, dont=True)

compare_models(data_states, [
    AvgModel(),
    AvgItemModel(),
    # EloTimeModel(),
    EloModel(alpha=1.2),
    # EloClusterModel(clusters=maps_continents_country),
    # EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    # EloClusterModel(clusters=maps_continents_country, separate=True),
    EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4),

    # EloTreeModel(clusters=maps_continents_country, local_update_boost=1),
], dont=True, resolution=False)

compare_models(data_europe, [
    AvgModel(),
    AvgItemModel(),
    # EloTimeModel(),
    EloModel(alpha=1.2, beta=.12),
    # EloClusterModel(clusters=europe_clusters),
    EloCorrModel(corr_place_weight=1., prior_weight=0.6),
    EloTreeModel(clusters=europe_clusters, local_update_boost=0.2),
    # EloTreeModel(clusters=ac, local_update_boost=0.4),
], dont=True)

group_rmse(data_europe, [
    AvgModel(),
    AvgItemModel(),
    EloTimeModel(),
    EloModel(alpha=1.2, beta=.12),
    EloClusterModel(clusters=europe_clusters),
    EloCorrModel(corr_place_weight=1., prior_weight=0.6),
    EloTreeModel(clusters=europe_clusters, local_update_boost=0.2),
    ], europe_clusters, dont=True)


compare_models(data_states, [
    AvgModel(),
    AvgItemModel(),
    EloModel(),
    EloModel(alpha=1.2),
    EloModel(alpha=1.2, random_factor="**"),
    EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    EloCorrModel(corr_place_weight=0.8, prior_weight=0.8, min_corr=200),
    # EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4),
    # EloTimeModel(),
], dont=True)

options_calibration(data_states, [
# options_rmse(data_states, [
# group_calibration(data_states, [
# group_rmse(data_states, [
#     AvgModel(),
#     AvgItemModel(),
#     EloTimeModel(),
    EloModel(alpha=1.2),
    EloModel(alpha=1.2, random_factor="**"),
# EloClusterModel(clusters=maps_continents_country),
    EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4),
], maps_continents_country, dont=True)


# elo_grid_search(data_states, run=False)
# elo_grid_search(data_states_old, run=False)
# elo_grid_search(data_europe, run=False)
# elo_grid_search(data_europe_old, run=False)
# elo_grid_search(data_cz_cities, run=False)
# elo_grid_search(data_cz_cities_old, run=False)
# elo_corr_grid_search(data_states, run=False)
# elo_corr_grid_search(data_states_old, run=False)
# elo_corr_grid_search(data_europe, run=False)
# elo_corr_grid_search(data_europe_old, run=False)
# elo_corr_grid_search(data_cz_cities, run=False)
# elo_corr_grid_search(data_cz_cities_old, run=False)

# elo_grid_search_gamma(data_states, run=False)

# corr_stats(data_europe, min_periods=1, test_dataset=True)

group_calibration(data_states, [
    # AvgItemModel(),
    EloModel(alpha=1.2),
    EloClusterModel(clusters=maps_continents_country),
    EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4),
], maps_continents_country, dont=True)


compare_models(data_states, [
    AvgModel(),
    AvgItemModel(),
    EloTimeModel(),
    EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4),
    EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4, version="global_and_cluster_at_once"),
    ], dont=True, resolution=False)


def skill_correlations():
    clusters = get_maps("data/")
    model = EloTreeModel(clusters=clusters, local_update_boost=0.5)
    Runner(data_all, model).run()
    global_skills = pd.Series(model.global_skill, index=model.global_skill.keys())

    skills = pd.DataFrame(data=global_skills, columns=["global"])
    for map in clusters.keys():
        local_skills = pd.Series(model.maps_skills[map], index=model.maps_skills[map].keys())
        skills[map] = local_skills

    skills = skills[(~skills.isnull()).sum(axis=1) > 6]

    for map in clusters.keys():
        if (~skills[map].isnull()).sum() == 0:
            continue

        plt.figure()
        tmp = skills[~skills[map].isnull()]
        plt.title(skills.corr().ix["global", map])
        print skills.corr().ix["global", map], map
        plt.plot(tmp["global"], tmp[map], ".")
        plt.ylabel(map)
        plt.xlabel("global")
        plt.savefig("results/tree-elo skill-corr {} filtered-6.png".format(map))

# skill_correlations()
#

corr = pd.DataFrame.from_csv("corr.tmp")
corr = corr.ix[1:, 1:]
corr[corr==1] = 0
print corr.std()

# plt.show()

if False:

    data_states1 = Data("data/geography-first-all-filtered-2.pd", train=0.3, force_train=1)
    data_states2 = Data("data/geography-first-all-filtered-2.pd", train=0.3, force_train=2)

    corr1, _, _ = compute_correlations(data_states1, min_periods=200)
    corr2, _, _ = compute_correlations(data_states2, min_periods=200)

    plt.title(sc.spearmanr(corr1.unstack(), corr2.unstack()))
    plt.plot(corr1.unstack(), corr2.unstack(), "o")

    plt.show()