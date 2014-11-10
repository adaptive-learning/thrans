from runner import *
from evaluator import *
from data.data import *
from models.elo_corr import *
import pylab as plt

data_states = Data("data/geography-first-states-filtered.pd", train=0.3)
data_europe = Data("data/geography-first-europe-filtered.pd", train=0.3)

# Runner(Data("data/geography-first-all.pd", test=False), EloModel()).run()
# Runner(Data("data/geography-first-all.pd", test=False), AvgModel()).run()
# Runner(Data("data/geography-first-all.pd", test=False), AvgItemModel()).run()

# Runner(Data("data/geography-first-states-filtered.pd", train=0.3), EloCorrModel()).run()

# run_all_models(data, run=True)


# experiment = Evaluator(data, EloCorrModel())

# experiment.evaluate()
# experiment.brier_graphs()
# print experiment

# compare_models(data_states, [AvgModel(), AvgItemModel(), EloModel(), EloCorrModel()])
#
#
# Evaluator(data, EloCorrModel()).brier_graphs(show=False)
# Evaluator(data, EloModel()).brier_graphs(show=False)
#

# elo_grid_search(data_states, run=False)
# elo_grid_search(data_europe, run=False)
elo_corr_grid_search(data_europe, run=False)



plt.show()
