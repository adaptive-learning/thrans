#!/usr/bin/python

from data import Data
from evaluator import *
from models import *
from model_corrElo import *
from model_clustElo import *

def compare_models(data, models, graphs=0):
    for i, m in enumerate(models):
        print str(m)
        m.process_data(data)
        e = Evaluator(m.log)
        RMSE = e.print_metrics()
        if i == 0:
            first = RMSE
        else:
            print "improvement {0:.2%}".format((first - RMSE) / first)
        if graphs:
            e.calibration_graphs()   
    if graphs:
        plt.show()

# places = []
# with open("clusters_continents.json") as f:
#     clusters = json.load(f)
#     for name, ps in clusters.items():
#         places += ps

# corr = pd.read_pickle("correlations.pd")
# places = corr.index
# svet = [60, 72, 80, 94, 99, 109, 114, 115, 118, 120, 126, 145, 153, 159, 176, 182, 184, 194, 195, 203, 216, 54, 63, 69, 77, 86, 89, 92, 104, 106, 111, 112, 119, 124, 130, 137, 148, 157, 160, 168, 169, 172, 173, 174, 197, 199, 201, 202, 209, 217, 219, 220, 841, 842, 56, 58, 66, 85, 93, 107, 110, 121, 122, 128, 141, 151, 155, 165, 167, 181, 186, 188, 192, 198, 205, 212, 213, 57, 71, 75, 82, 87, 96, 98, 102, 125, 129, 131, 136, 149, 150, 152, 156, 158, 161, 162, 163, 175, 187, 207, 208, 214, 215, 218, 222, 223, 224]
places = []
maps = json.load(open("maps2.json"))
for _, ps in maps.items():
    places += ps
data = Data("data-first.csv", places=places)
# data1 = Data("data-first-0.5-1.csv", places=None)
# data2 = Data("data-first-0.5-2.csv", places=None)

# compare_models([ModelStupid(), ModelElo(), ModelEloCorr(places=places)], 0, 0, places=places)
compare_models(data, [
                   ModelElo(alpha=0.8, beta=0.05),
                   ModelElo_clust2(alpha=0.8, beta=0.05, corr_place_weight=0.05, prior_weight=1),
                   ModelElo_clust2(alpha=0.8, beta=0.05, corr_place_weight=0.1, prior_weight=1),
                   ModelElo_clust2(alpha=0.8, beta=0.05, corr_place_weight=0.5, prior_weight=1),
                   # ModelStupid(),
                   # ModelEloCorr(places=europe, alpha=0.4, beta=0.05, corr_places_count=0, corr_place_weight=2.4, prior_weight=0),
                   # ModelCorrElo(data1, alpha=0.4, corr_place_weight=2.4, prior_weight=0),
                   # ModelCorrElo(data1, alpha=0.8, corr_place_weight=2.4, prior_weight=0),
                   # ModelCorrElo(data1, alpha=0.8, corr_place_weight=2.4, prior_weight=1),
                   # ModelCorrElo(data1, alpha=0.8, corr_place_weight=1.4, prior_weight=1),
                   # ModelCorrElo(data1, alpha=0.8, corr_place_weight=1.4, prior_weight=0.5),
                   # ModelEloClust(alpha=0.5, beta=0.05, clusters=clusters),
                   # ModelEloClust(alpha=0, beta=0.05, clusters=clusters),
               ], graphs=0)
