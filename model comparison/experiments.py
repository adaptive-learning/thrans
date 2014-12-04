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

data_states = Data("data/geography-first-states-filtered.pd", train=0.3)
data_europe = Data("data/geography-first-europe-filtered.pd", train=0.3)
data_cz_cities = Data("data/geography-first-cz_city-filtered.pd", train=0.3)
maps_continents_country = get_continents_country_maps("data/")

# Runner(Data("data/geography-first-all.pd", test=False), EloModel()).run()
# Runner(Data("data/geography-first-all.pd", test=False), AvgModel()).run()
# Runner(Data("data/geography-first-all.pd", test=False), AvgItemModel()).run()

# model = EloCorrModel()
# Runner(data_states, model).run()
# experiment = Evaluator(data_states, model)
# experiment.evaluate()


# run_all_models(data, run=True)

# experiment = Evaluator(data_states, EloClusterModel(clusters=maps_continents_country))
# experiment = Evaluator(data_cz_cities, EloModel(alpha=0.6, beta=0.08))
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
compare_models(data_states, [
    AvgModel(),
    AvgItemModel(),
    EloModel(alpha=1.2),
    EloTimeModel(),
], dont=True)

compare_models(data_states, [
    AvgModel(),
    AvgItemModel(),
    EloTimeModel(),
    EloModel(alpha=1.2),
    EloClusterModel(clusters=maps_continents_country),
    EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    # EloClusterModel(clusters=maps_continents_country, separate=True),
    EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4),
    # EloTreeModel(clusters=maps_continents_country, local_update_boost=1),
], dont=True)

compare_models(data_europe, [
    AvgModel(),
    AvgItemModel(),
    EloTimeModel(),
    EloModel(alpha=1.2, beta=.12),
    EloClusterModel(clusters=europe_clusters),
    EloCorrModel(corr_place_weight=1., prior_weight=0.6),
    EloTreeModel(clusters=europe_clusters, local_update_boost=0.2),
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


group_rmse(data_states, [
    AvgModel(),
    AvgItemModel(),
    EloTimeModel(),
    EloModel(alpha=1.2),
    EloClusterModel(clusters=maps_continents_country),
    EloCorrModel(corr_place_weight=0.8, prior_weight=0.8),
    EloTreeModel(clusters=maps_continents_country, local_update_boost=0.4),
], maps_continents_country, dont=True)

# elo_grid_search(data_states, run=False)
# elo_grid_search(data_europe, run=False)
# elo_grid_search(data_cz_cities, run=False)
# elo_corr_grid_search(data_states, run=False)
# elo_corr_grid_search(data_europe, run=False)
# elo_corr_grid_search(data_cz_cities, run=False)

corr_stats(data_europe, min_periods=1, test_dataset=True)


plt.show()

# TODO sepsat to - related work; vzorezcky
# TODO correlace - jake jsou;
# TODO Elo corr - jak to delat jinak
# TODO hiearchucal skills - jak to delaji jinde, jak to delat jinak
# TODO alpha, beta, pocitat z train_setu