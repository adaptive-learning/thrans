import sys
sys.path.append("../Algoritmy")
from runner import *
from evaluator import *
from data.data import *
from models.elo_corr import *
from models.elo_tree import *
from clusters import *
import pylab as plt
import networkx as nx
from networkx_viewer import Viewer
from matplotlib import cm

colors = ["blue", "red", "green", "cyan", "purple", "black", "orange", "magenta", "gray", "yellow"]

places_all = places.from_csv("data/raw data/geography.place.csv", "data/raw data/geography.placerelation.csv","data/raw data/geography.placerelation_related_places.csv")
places_all.set_index(places_all["id"], inplace=True)

data_all = Data("data/geography-first-all-2.pd", train=0.3)
data_states = Data("data/geography-first-states-filtered-2.pd", train=0.3)
data_europe = Data("data/geography-first-europe-filtered-2.pd", train=0.3)
data_off = Data("data/geographyOF-first.pd", train=0.3, force_train=42)
data_off_europe = Data("data/geographyOF-first-europe.pd", train=0.3, force_train=42)
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

compare_models(data_off_europe, [
    # AvgModel(),
    # AvgItemModel(),
    EloModel(beta=0.06),
    # EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloTreeModel(clusters=europe_concepts, local_update_boost=0.25),
    EloTreeModel(clusters=correct_cluster(data_off_europe, europe_concepts), local_update_boost=0.25),
    EloTreeModel(clusters=europe_concepts2, local_update_boost=0.25),
    EloTreeModel(clusters=correct_cluster(data_off_europe, europe_concepts2), local_update_boost=0.25),
    EloTreeModel(clusters=compute_clusters(data_off_europe, 2), local_update_boost=0.25),
    EloTreeModel(clusters=compute_clusters(data_off_europe, 3), local_update_boost=0.25),
    EloTreeModel(clusters=compute_clusters(data_off_europe, 5), local_update_boost=0.25),
    ], dont=0, resolution=True, evaluate=False, diff_to=0.38258)

compare_models(data_off, [
    # AvgModel(),
    # AvgItemModel(),
    EloModel(beta=0.06),
    # EloCorrModel(corr_place_weight=1., prior_weight=0.8, min_corr=200),
    EloTreeModel(clusters=get_maps("data/"), local_update_boost=0.5),
    EloTreeModel(clusters=get_maps("data/", just_maps=True), local_update_boost=0.5),
    EloTreeModel(clusters=get_maps("data/", just_types=True), local_update_boost=0.5),
    EloTreeModel(clusters=correct_cluster(data_off, get_maps("data/")), local_update_boost=0.5),
    EloTreeModel(clusters=correct_cluster(data_off, get_maps("data/", just_types=True)), local_update_boost=0.5),
    EloTreeModel(clusters=correct_cluster(data_off, get_maps("data/", just_maps=True)), local_update_boost=0.5),
    # EloTreeModel(clusters=one_concept, local_update_boost=0.5),
    # EloTreeModel(clusters=random_concepts(one_concept["all"], 2), local_update_boost=0.5),
    # EloTreeModel(clusters=random_concepts(one_concept["all"], 4), local_update_boost=0.5),
    EloTreeModel(clusters=compute_clusters(data_off, 5), local_update_boost=0.5),
    EloTreeModel(clusters=compute_clusters(data_off, 20), local_update_boost=0.5),
    EloTreeModel(clusters=compute_clusters(data_off, 50), local_update_boost=0.5),
    ], dont=1, resolution=True, diff_to=0.4142, evaluate=False)

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

if False:
    data = data_off_europe
    # filename = "data/{}.corr.pd".format(sha1(str(data)).hexdigest()[:10])
    filename = "data/{}{}.corr.pd".format(sha1(str(data)).hexdigest()[:10], " min_corr: " + str(200))
    corr = pd.read_pickle(filename)
    graph = nx.from_numpy_matrix(corr.values)
    names = {}
    for i, id in enumerate(corr.columns):
        names[i] = places_all.loc[id, "name_en"]


    names = {i: n.decode("utf-8").encode("ascii", "ignore") for i, n in names.items()}

    graph = nx.relabel_nodes(graph, names)
    # graph2 = nx.Graph([(u, v, d) for u, v, d in graph.edges(data=True) if d['weight'] > 0.4])
    # graph2 = nx.Graph([(u, v, d) for u, v, d in graph.edges(data=Truxe) if 1 > d['weight'] > 0.4])
    # graph2 = nx.Graph([(u, v, d) for u, v, d in graph.edges(data=True) if d['weight'] > sorted([d["weight"] for d in graph[u].values()], reverse=True)[3]])

    graph2 = nx.Graph()

    weights = set()
    for node in graph.nodes():
        for edge in sorted(list(graph[node].items()), key=lambda d: -d[1]["weight"])[:3]:
            graph2.add_edge(node, edge[0], edge[1])
            weights.add(edge[1]["weight"])

    edges = graph2.edges()
    # colors = [graph2[u][v]['color'] for u,v in edges]
    widths = [(graph2[u][v]['weight'] - min(weights) + 0.02) * 15 for u, v in edges]
    labels = {(u, v): str(graph2[u][v]["weight"])[:4] for u, v in edges}

    pos = nx.graphviz_layout(graph2)
    nx.draw(graph2, pos=pos, edgelist=edges, width=widths)
    nx.draw_networkx_labels(graph2, pos={k: (v[0], v[1]-15) for k, v in pos.items()})
    nx.draw_networkx_edge_labels(graph2, pos=pos, edge_labels=labels, font_size=8)
    print pos

    # app = Viewer(graph2)
    # app.mainloop()


# plt.savefig("results/europe-graph.svg")
plt.show()