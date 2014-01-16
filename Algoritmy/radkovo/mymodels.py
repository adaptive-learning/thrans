import numpy as np
import scipy as sp
import scipy.stats
import math
from collections import deque

# pro kratsi zapis obecne pouzito:
#    s = student
#    p = problem

def linear_sum(l):
    s, w = 0, 0
    for i in range(len(l)):
        s += l[i] * (i+1)
        w += i+1
    return float(s) / w

class Model:
    name = "abstract"
    
    def __init__(self, data):
        self.data = data
        self.par = {}
        self.gd_default_iter = 1
        self.gd_default_gamma = 0.005

    def predict(self): return 0
    
    def normalize_param(self): pass

    def local_improvement(self, s, p, err): pass

    def error(self, s, p):
        return self.data.times[s][p] - self.predict(s, p)
    
    def compute_prediction_matrix(self):
        self.pred_times = np.zeros((self.data.students, self.data.problems))
        for s in range(self.data.students):
            for p in range(self.data.problems):
                self.pred_times[s][p] = self.predict(s,p)    
    
class ModelMean(Model):
    name = "ModelMean"
    
    def __init__(self, data):
        Model.__init__(self, data)
        self.par['m'] = self.data.problem_mean_time
        
    def predict(self, s, p):                           
        return self.par['m'][p]

class ModelBaseline(Model):
    name = "ModelBaseline"
    
    def __init__(self, data):
        Model.__init__(self, data)
        self.par['m'] = np.copy(self.data.problem_mean_time)
        self.par['skill'] = - np.copy(self.data.student_mean_deviation)
        
    def predict(self, s, p):                           
        return self.par['m'][p] - self.par['skill'][s]

    # simuluje pruchod studenta a prubezne predikce
    # tady uvazuji volitelne i discounting, u zakladniho init/predict zatim ne
    def prediction_sequence_residuals(self, problem_ids, student_times_seq, discount = 1):
        residuals = np.zeros_like(student_times_seq)
        dev_sum, skill, weight = 0, 0, 0
        for i in range(len(problem_ids)):
            p = problem_ids[i]
            residuals[i] = student_times_seq[i] - (self.par['m'][p] - skill)
            dev_sum = discount * dev_sum - (student_times_seq[i] - self.par['m'][p])
            weight = discount * weight + 1
            skill = dev_sum / weight     
        return residuals        
    
    # linear version remake
    def prediction_sequence_residuals_window(self, problem_ids, student_times_seq, window_size = float('inf')):
        residuals = np.zeros_like(student_times_seq)
        skill = 0
        devs = deque([])
        for i in range(len(problem_ids)):
            p = problem_ids[i]
            residuals[i] = student_times_seq[i] - (self.par['m'][p] - skill)
            devs.append(student_times_seq[i] - self.par['m'][p])
            if len(devs) > window_size:
                devs.popleft()
#            skill = - float(sum(devs)) / len(devs)
            skill = - linear_sum(devs)
        return residuals        

    
class Model11(Model):
    name = "Model11"
    
    def __init__(self, data):
        Model.__init__(self, data)
        self.par['skill'] = - np.copy(self.data.student_mean_deviation)
        self.par['b'] = np.copy(self.data.problem_mean_time)
        self.gd_default_iter = 3

    def normalize_param(self):
        mean_skill = np.mean(self.par['skill'])
        self.par['skill'] = self.par['skill'] - mean_skill
        self.par['b'] = self.par['b'] - mean_skill
        
    def predict(self, s, p):                           
        return self.par['b'][p] - self.par['skill'][s]
    
    def local_improvement(self, s, p, err):
        self.par['skill'][s] += self.gamma * err * (-1)
        self.par['b'][p]     += self.gamma * err
    

class Model12(Model):                           
    name = "Model12"
    
    def __init__(self, data):
        Model.__init__(self, data)
        self.par['skill'] = - np.copy(self.data.student_mean_deviation)
        self.par['b'] = np.copy(self.data.problem_mean_time)
        self.par['a'] = -1 * np.ones(self.data.problems)
        self.gd_default_iter = 10

    def normalize_param(self):
        mean_skill = np.mean(self.par['skill'])
        self.par['b'] = self.par['b'] - mean_skill
        self.par['skill'] = self.par['skill'] - mean_skill
        mean_a = np.mean(self.par['a'])
        self.par['a'] = - self.par['a'] / mean_a
        self.par['skill'] = - self.par['skill'] * mean_a
                                                        
    def predict(self, s, p):
        return self.par['b'][p] + self.par['a'][p] * self.par['skill'][s]

    def local_improvement(self, s, p, err):
        orig_skill = self.par['skill'][s]
        self.par['skill'][s] += self.gamma * err * self.par['a'][p]
        self.par['b'][p]     += self.gamma * err
        self.par['a'][p]     += self.gamma * err * orig_skill

    def prediction_sequence_residuals(self, problem_ids, student_times_seq, discount = 1):
        residuals = np.zeros_like(student_times_seq)
        dev_sum, skill, weight = 0, 0, 0
        for i in range(len(problem_ids)):
            p = problem_ids[i]
            residuals[i] = student_times_seq[i] - (self.par['b'][p] + self.par['a'][p] * skill)
            dev_sum = discount * dev_sum + (student_times_seq[i] - self.par['b'][p])/self.par['a'][p]
            weight = discount * weight + 1
            skill = dev_sum / weight     
        return residuals        

        
class Model12var(Model12):                           
    name = "Model12var"
    
    def __init__(self, data):
        Model12.__init__(self, data)
        self.par['c'] = sp.stats.nanstd(self.data.time_dev, axis = 0) ** 2 / 2
        self.par['svar'] = sp.stats.nanstd(self.data.time_dev, axis = 1) ** 2 / 2
        self.gd_default_iter = 30
        self.gd_default_gamma = 0.004
        
    # normalize a predict dedime od Model12
        
    def local_improvement(self, s, p, err):
        var = self.par['c'][p] + (self.par['a'][p]**2) * self.par['svar'][s]
        orig_skill = self.par['skill'][s]
        orig_a = self.par['a'][p]

        self.par['skill'][s] += self.gamma * (err/var) * self.par['a'][p]
        self.par['b'][p]  += self.gamma * (err/var)
        aps = self.par['a'][p]*self.par['svar'][s]
        self.par['a'][p]  += self.gamma * ((err/var) * (orig_skill + aps*err/var) - aps/var)
        self.par['c'][p] = max(self.par['c'][p] + self.gamma * (err**2 - var) / (2 * var**2), 0.05)
        self.par['svar'][s] = max(self.par['svar'][s] + self.gamma * orig_a**2 * (err**2 - var) / (2 * var**2), 0.05)
        

class Model22L(Model12):                           
    name = "Model22L"
    
    def __init__(self, data, order_fun = lambda x: math.log(x+1,2), delta_init = 0.3, reg_lambda = 0):
        Model12.__init__(self, data)
        self.order_fun = order_fun
        self.par['delta'] = np.ones(self.data.students) * delta_init
        self.reg_lambda = reg_lambda
        self.gd_default_iter = 60
        self.gd_default_gamma = 0.004

    def normalize_param(self):
        mean_skill = np.mean(self.par['skill'])
        self.par['b'] = self.par['b'] - mean_skill
        self.par['skill'] = self.par['skill'] - mean_skill
        mean_a = np.mean(self.par['a'])
        self.par['a'] = - self.par['a'] / mean_a
        self.par['skill'] = - self.par['skill'] * mean_a
        self.par['delta'] = - self.par['delta'] * mean_a       

    # trochu nesystemove, protoze vypocet vyuziva order, ktery neni vstupem...
    def predict(self, s, p):
        return self.par['b'][p] + self.par['a'][p] * (self.par['skill'][s] + self.par['delta'][s] * self.order_fun(self.data.order[s][p]))
        
    # def local_improvement(self, s, p, err):
    #     orig_skill_sum = self.par['skill'][s]  + self.par['delta'][s] * self.order_fun(self.data.order[s][p])
    #     self.par['skill'][s] += self.gamma * err * self.par['a'][p]
    #     self.par['delta'][s] += self.gamma * err * self.par['a'][p] * self.order_fun(self.data.order[s][p])
    #     self.par['b'][p]     += self.gamma * err
    #     self.par['a'][p]     += self.gamma * err * orig_skill_sum

    # s regularizaci
    def local_improvement(self, s, p, err):
        orig_skill_sum = self.par['skill'][s]  + self.par['delta'][s] * self.order_fun(self.data.order[s][p])
#        self.par['skill'][s] = self.par['skill'][s] * (1-self.reg_lambda*self.gamma) + self.gamma * err * self.par['a'][p]
        self.par['delta'][s] = self.par['delta'][s] * (1-self.reg_lambda*self.gamma) +  self.gamma * err * self.par['a'][p] * self.order_fun(self.data.order[s][p])
        self.par['b'][p]     += self.gamma * err
        self.par['a'][p]     += self.gamma * err * orig_skill_sum


class Model22Lreg(Model22L):
    name = "Model22Lreg"
    
    def __init__(self, data):
        Model22L.__init__(self, data, reg_lambda = 3.0)
        
class Model22Lsqrt(Model22L):
    name = "Model22Lsqrt"
    
    def __init__(self, data):
        Model22L.__init__(self, data, math.sqrt)
    
