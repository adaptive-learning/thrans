import random
import math
from helper import *
import scipy as sp
from scipy.stats import spearmanr
import pylab as plt

########## vlastni gradient descent ############

def gradient_descent_one_iter(model):
    student_list = range(model.data.students)
    random.shuffle(student_list)
    problem_list = range(model.data.problems)
    random.shuffle(problem_list)
    for s in student_list:
        for p in problem_list:
            if not math.isnan(model.data.times[s][p]):
                err = model.error(s,p)
                model.local_improvement(s, p, err)
    
def stochastic_gradient_descent(model, observer = None,
                                iterations = 0, gamma = 0):
    if gamma == 0: gamma = model.gd_default_gamma
    if iterations == 0: iterations = model.gd_default_iter
    model.gamma = gamma
    for i in range(iterations):
        gradient_descent_one_iter(model)
        model.normalize_param()
        if observer: observer.log(i)

####################################################        
# logovani prubehu grad. descentu a vykreslovani grafu o prubehu
class GDobserver:
    def __init__(self, model, test=None, correlations = False):
        self.model = model
        self.test_rmse = []
        self.train_rmse = []
        self.test = test
        self.correlations = correlations
        if correlations:
            self.r = {}
            for param in model.par:
                if param in model.data.par:
                    self.r[param] = []
        self.verbose = 0
        
    def log(self, i):
        self.model.compute_prediction_matrix()
        self.train_rmse.append(prediction_rmse(self.model.pred_times,
                                               self.model.data))
        if self.test:
            self.test_rmse.append(prediction_rmse(self.model.pred_times,
                                                  self.test))
        else:
            self.test_rmse.append(0)
        if self.correlations:
            for param in self.r:
                self.r[param].append(spearmanr(self.model.par[param],
                                               self.model.data.par[param])[0])
        if self.verbose:
            print i, round(self.train_rmse[-1],3), round(self.test_rmse[-1],3)

    def final_test_rmse(self):
        return self.test_rmse[-1]
            
    def plot_rmses(self):
        plt.figure()
        legend = []
        plt.plot(self.train_rmse)
        legend.append('train')
        if self.test_rmse[0]:
            plt.plot(self.test_rmse)
            legend.append('test')
        plt.legend(legend)

    def plot_r(self):
        plt.figure()
        legend = []
        for param in self.r:
            plt.plot(self.r[param])
            legend.append(param)
        plt.legend(legend)        
        
