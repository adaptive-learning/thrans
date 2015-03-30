import math
import numpy as np
import sklearn.metrics as met
import pylab as plt


SIZE = 10**5
PROB = np.random.rand(SIZE)

observed = (np.random.rand(SIZE) < PROB) * 1
real = PROB

noises = np.arange(0.001, 0.2, 0.01)
rmses = []

for noise in noises:
    predicted = map(lambda x: max(min(1, x), 0), real + np.random.normal(0, noise, SIZE))
    rmses.append(math.sqrt(met.mean_squared_error(observed, predicted)))

plt.plot(noises, rmses)
plt.show()