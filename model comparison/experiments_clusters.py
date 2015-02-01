import sys
sys.path.append("../Algoritmy")
from runner import *
from evaluator import *
from data.data import *
from models.elo_corr import *
from models.elo_tree import *
from clusters import *
import pylab as plt

colors = ["blue", "red", "green", "cyan", "purple", "black", "orange", "magenta", "gray", "yellow"]

places_all = places.from_csv("data/raw data/geography.place.csv", "data/raw data/geography.placerelation.csv","data/raw data/geography.placerelation_related_places.csv")
places_all.set_index(places_all["id"], inplace=True)

data_all = Data("data/geography-first-all-2.pd", train=0.3)
data_states = Data("data/geography-first-states-filtered-2.pd", train=0.3)
data_europe = Data("data/geography-first-europe-filtered-2.pd", train=0.3)
maps_continents_country = get_continents_country_maps("data/")
maps_all = get_maps("data/")
one_concept = {"all": reduce(lambda a, b: a + maps_all[b], maps_all, [])}
europe_concepts = json.load(open("data/europe_concepts.json"))
europe_concepts2 = json.load(open("data/europe_concepts2.json"))


# compute_clusters(data_all, 10)


# correct_cluster(data_states, maps_continents_country, run=True, print_diffs=True)
# correct_cluster(data_all, get_maps("data/"), run=True, print_diffs=True)
# correct_cluster(data_europe, europe_concepts, run=True, print_diffs=True)
# correct_cluster(data_europe, europe_concepts2, run=True, print_diffs=True)


# cluster_optim(data_europe, [europe_concepts, europe_concepts2],  range(2, 6, 1), local_update_boost=0.25)
# cluster_optim(data_states, [maps_continents_country],  range(2, 30, 1))
# cluster_optim(data_all, [get_maps("data/"), get_maps("data/", just_types=True), get_maps("data/", just_maps=True)], range(2, 100, 2))

# compare_brier_curve(data_europe, EloModel(beta=0.06), EloTreeModel(clusters=compute_clusters(data_europe, 3), local_update_boost=0.25),)
tmp_maps = get_maps("data/")
group_calibration(data_all, [
# group_rmse(data_all, [
    EloTreeModel(clusters=tmp_maps, local_update_boost=0.5),
    EloTreeModel(clusters=correct_cluster(data_all, tmp_maps), local_update_boost=0.5)
    ], tmp_maps, dont=1)

compare_models(data_europe, [
    # AvgModel(),
    # AvgItemModel(),
    EloModel(beta=0.06),
    # EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloTreeModel(clusters=europe_concepts, local_update_boost=0.25),
    EloTreeModel(clusters=correct_cluster(data_europe, europe_concepts), local_update_boost=0.25),
    EloTreeModel(clusters=europe_concepts2, local_update_boost=0.25),
    EloTreeModel(clusters=correct_cluster(data_europe, europe_concepts2), local_update_boost=0.25),
    EloTreeModel(clusters=compute_clusters(data_europe, 2), local_update_boost=0.25),
    EloTreeModel(clusters=compute_clusters(data_europe, 3), local_update_boost=0.25),
    EloTreeModel(clusters=compute_clusters(data_europe, 5), local_update_boost=0.25),
    ], dont=1, resolution=True, evaluate=False, diff_to=0.37158)

compare_models(data_europe, [
    # AvgModel(),
    # AvgItemModel(),
    EloModel(beta=0.06),
    # EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),
    EloTreeModel(clusters=get_maps("data/", just_maps=True), local_update_boost=0.5),
    EloTreeModel(clusters=get_maps("data/", just_types=True), local_update_boost=0.5),
    EloTreeModel(clusters=correct_cluster(data_all, get_maps("data/")), local_update_boost=0.5),
    EloTreeModel(clusters=correct_cluster(data_all, get_maps("data/", just_types=True)), local_update_boost=0.5),
    EloTreeModel(clusters=correct_cluster(data_all, get_maps("data/", just_maps=True)), local_update_boost=0.5),
    # EloTreeModel(clusters=one_concept, local_update_boost=0.5),
    # EloTreeModel(clusters=random_concepts(one_concept["all"], 2), local_update_boost=0.5),
    # EloTreeModel(clusters=random_concepts(one_concept["all"], 4), local_update_boost=0.5),
    EloTreeModel(clusters=compute_clusters(data_all, 5), local_update_boost=0.5),
    EloTreeModel(clusters=compute_clusters(data_all, 20), local_update_boost=0.5),
    EloTreeModel(clusters=compute_clusters(data_all, 50), local_update_boost=0.5),
], dont=1, resolution=True, diff_to=0.40759, evaluate=True)

compare_models(data_states, [
    # AvgModel(),
    # AvgItemModel(),
    EloModel(),
    EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloTreeModel(clusters=maps_continents_country, local_update_boost=0.5),
    EloTreeModel(clusters=compute_clusters(data_states, 7), local_update_boost=0.5),
], dont=1, resolution=True)

if False:
    params = np.arange(0, 1, 0.05)
    results = []
    for param in params:
        model = EloTreeModel(clusters=europe_concepts, local_update_boost=param)
        results.append(Evaluator(data_europe, model).get_report()["rmse"])

    plt.figure()
    plt.plot(params, results, '-')

if False:
    params =np.array([0.00001, 0.0001, 0.001, 0.005, 0.01, 0.1, 0.5, 1, 2, 10, 1000])
    results = []
    for param in params:
        model = EloTreeModel(clusters=correct_cluster(data_all, get_maps("data/"), regularization=param), local_update_boost=0.5)
        results.append(Evaluator(data_all, model).get_report()["rmse"])

    plt.figure()
    plt.plot(np.log(params), results, '-')

# plt.savefig("results/tmp2.png")
plt.show()