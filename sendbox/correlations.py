import scipy.stats as sc
import numpy as np


x = np.random.random(1000)

print sc.spearmanr(x, x>0.1)[0]
print sc.pearsonr(x, x>0.1)[0]

[a,b] = [1,2]

print a, b
