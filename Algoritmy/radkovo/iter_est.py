import math
import numpy as np
import pylab as plt
from helper import *

def linregression(x, y):
    a = np.matrix([ x, np.ones(len(x)) ]).transpose()
    return np.linalg.lstsq(a,y)[0]

####### Model 12 ##########

def improve_problem_par(model):
    for p in range(model.data.problems):
        x, y = [], []
        for s in range(model.data.students):
            if not math.isnan(model.data.times[s][p]):
                x.append(model.par['skill'][s])
                y.append(model.data.times[s][p])
        model.par['a'][p], model.par['b'][p] = linregression(x,y)
        
def improve_student_par(model):
    for s in range(model.data.students):
        dev_sum = 0.0
        count = 0
        for p in range(model.data.problems):
            if not math.isnan(model.data.times[s][p]):
                dev_sum += (model.data.times[s][p] - model.par['b'][p]) / model.par['a'][p]
                count += 1
        model.par['skill'][s] = dev_sum / count

def iterative_estimation(model, iterations = 6):
    for i in range(iterations):
        improve_problem_par(model)
        improve_student_par(model)
        model.normalize_param()

##### Model learning ##

def improve_problem_parL(model):
    for p in range(model.data.problems):
        x, y = [], []
        for s in range(model.data.students):
            if not math.isnan(model.data.times[s][p]):
                x.append(model.par['skill'][s] + model.par['delta'][s] * model.order_fun(model.data.order[s][p]))
                y.append(model.data.times[s][p])
        model.par['a'][p], model.par['b'][p] = linregression(x,y)
        
def improve_student_parL(model):
    for s in range(model.data.students):
        x, y = [], []
        for p in range(model.data.problems):
            if not math.isnan(model.data.times[s][p]):
                x.append(model.order_fun(model.data.order[s][p]))
                y.append((model.data.times[s][p] - model.par['b'][p])/model.par['a'][p])
        model.par['delta'][s], model.par['skill'][s] = linregression(x,y)

        
def iterative_estimationL(model, iterations = 6):
    for i in range(iterations):
        improve_problem_parL(model)
        improve_student_parL(model)
        model.normalize_param()
        
