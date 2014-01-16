import math
import numpy as np
import pylab as plt
from scipy.stats import spearmanr

# drobne pomocne funkce

################ generovani a dilci pomocne ##############

def nans(x, y):
    out = np.empty((x,y))
    out.fill(np.nan)
    return out

def get_problem_id_in_order(sorder, solved):
    problem_ids = [ 0 for _ in range(solved) ]
    for i in range(len(sorder)):
        if not math.isnan(sorder[i]):
            if sorder[i] <= solved:
                problem_ids[int(sorder[i])-1] = i
    return problem_ids


################# vypocty ################

def rmse(X,Y):
    ssum = 0.0
    for x,y in zip(X,Y):
        ssum += (x-y)**2;
    return math.sqrt(ssum/len(X))

def prediction_rmse(pred_matrix, test):
    suma = np.nansum((test.times - pred_matrix) ** 2)
    return math.sqrt( suma / test.valid_times)    

#normalized varianta
def nrmse(X,Y):
    ssum = 0.0
    for x,y in zip(X,Y):
        ssum += (x-y)**2;
    return math.sqrt(ssum/len(X)) / (max(X) - min(X))    


def correlation_common(x, y):
    xx, yy = [], []
    for i in range(len(x)):
        if not math.isnan(x[i]) and not math.isnan(y[i]):
            xx.append(x[i])
            yy.append(y[i])
    return spearmanr(xx,yy)[0]        

def common_students(x, y): # stupid quick hack, jde urcite lip pres numpy
    count = 0
    for i in range(len(x)):
        if not math.isnan(x[i]) and not math.isnan(y[i]):
            count += 1
    return count 


############### vykreslovani ##################

def plot_scatter_line(p1, p2, desc, symbol = 'o'):
    plt.figure()
    plt.title(desc)
    plt.plot(p1, p2, symbol)
    plt.plot((min(p1), max(p1)), (min(p1),max(p1)))
    print desc, "spearman", round(spearmanr(p1,p2)[0],2)

def plot_scatter(p1, p2, labelx, labely, symbol = 'o'):
    plt.figure()
    plt.plot(p1, p2, symbol)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    print labelx, labely, "spearman", round(spearmanr(p1,p2)[0],2)
    
def plot_hist(p, desc = "unknown entity"):
    plt.figure()
    plt.title(desc)
    plt.hist(p)
    print desc, "mean:", round(np.mean(p),2)
    print desc, "std:", round(np.std(p),2)
    
def plot_line(x, a, b, c = 0):
    minx = min(x) - 0.2
    maxx = max(x) + 0.2
    if c == 0:
        plt.plot((minx, maxx), (a*minx +b, a*maxx+b))
    else:
        plt.plot((minx, maxx), (a*minx +b, a*maxx+b), c = c)
    
#################### logovani opakovanych behu ################

class MultipleRunLogger:
    
    def __init__(self, verbose = 0):
        self.row_names = []
        self.col_names = [] 
        self.data = {}
        self.verbose = verbose

    def log(self, r, c, value):
        if not r in self.row_names:
            self.row_names.append(r)
        if not c in self.col_names:
            self.col_names.append(c)
        if not (r,c) in self.data:
            self.data[r,c] = []        
        self.data[r,c].append(value)
        if self.verbose:
            print r, c, len(self.data[r,c]), value
        
    def print_table(self, sep = "\t", line_end = ""):
        print
        print sep, sep.join(self.col_names), line_end
        for r in self.row_names:
            print r, 
            for c in self.col_names:
                if len(self.data[r,c]) == 0:
                    print sep, "-", 
                else:
                    print sep, round(float(sum(self.data[r,c])) / len(self.data[r,c]), 3), 
            print line_end
        print
        
            
