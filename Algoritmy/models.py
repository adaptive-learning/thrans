import numpy as np
import pylab as plt
import random
import math

class Model:
    def __init__(self, skill_pars, problem_pars, data):
        self.data = data
        self.problem_pars = problem_pars
        self.skill_pars = skill_pars
        self.ppars = np.zeros((problem_pars, data.problems)) 
        self.skills = np.zeros((skill_pars, data.students))
        self.set_init_parameters()
        
    def gradient_descent_one_iter(self):
        student_list = range(self.data.students)
        random.shuffle(student_list)
        problem_list = range(self.data.problems)
        random.shuffle(problem_list)
        for s in student_list:
            for p in problem_list:
                if not np.isnan(self.data.times[s][p]):
                    err = self.error(s,p)
                    self.local_improvement(s, p, err)
    
    def stochastic_gradient_descent(self, repeat = 40, gamma = 0.005, test = None, verbose = 0):
        if test == None: test = self.data # trochu hack na zjednoduseni impl.
        self.gamma = gamma
        test_rmse = []
        train_rmse = []
        for i in range(repeat):
            self.gradient_descent_one_iter()
            self.normalize_param()
            rmses = self.prediction_rmses(test)
            train_rmse.append(rmses[0])
            test_rmse.append(rmses[1])
            if verbose:
                print i, train_rmse[i], test_rmse[i]
        return (train_rmse, test_rmse)
    

    def prediction_matrix(self):
        pred_times = np.zeros((self.data.students, self.data.problems))        
        for s in range(self.data.students):
            for p in range(self.data.problems):
                pred_times[s][p] = self.predict(s,p)
        return pred_times
                
    def error(self, student, problem):
        return self.data.times[student][problem] - self.predict(student, problem)

    # optimalizace na rychlost, abych to neprochazel 3x
    def prediction_rmses(self, test):
        suma_train = 0
        suma_test = 0
        for s in range(self.data.students):
            for p in range(self.data.problems):
                if not math.isnan(self.data.times[s][p]):
                    suma_train += (self.data.times[s][p] - self.predict(s,p))**2
                if not math.isnan(test.times[s][p]):
                    suma_test += (test.times[s][p] - self.predict(s,p))**2
        return (math.sqrt(suma_train / self.data.valid_times),
                math.sqrt(suma_test / test.valid_times))
    
    # predpoklada, ze test_data maji stejne dimenze
    def prediction_rmse(self, test_data = None):
        if test_data == None: test_data = self.data
        suma = np.nansum((test_data.times - self.prediction_matrix()) ** 2)
        return math.sqrt( suma / test_data.valid_times)

    def problem_sd(self, test_data = None):
        if test_data == None: test_data = self.data
        suma = np.nansum((test_data.times - self.prediction_matrix()) ** 2, axis = 0)
        return np.sqrt( suma / test_data.problem_solved )        
    
    def predict(self):
        return 0
    
    def normalize_param(self):
        pass

    def set_init_parameters(self):
        pass

    def local_improvement(self, student, problem, err):
        pass
    
class Model2(Model):                           
    # skill[0] - zakladni skill
    # ppar[0]  - zakladni obtiznost problemu
    # ppar[1]  - diskriminace

    def __init__(self, data):
        Model.__init__(self, 1, 2, data)

    def set_init_parameters(self):
        self.skills[0] = - self.data.student_mean_deviation
        self.ppars[0] = self.data.problem_mean_time
        self.ppars[1] = -1 * np.ones(self.data.problems)

    def normalize_param(self):
        mskill = np.mean(self.skills[0])
        self.skills[0] = self.skills[0] - mskill
        self.ppars[0] = self.ppars[0] - mskill
        mpar1 = np.mean(self.ppars[1])
        self.ppars[1] = - self.ppars[1] / mpar1
        self.skills[0] = - self.skills[0] * mpar1
        
    def predict(self, s, p):
        return self.ppars[0][p] + self.ppars[1][p] * self.skills[0][s]

    def local_improvement(self, s, p, err):
        orig_skill = self.skills[0][s]
        self.skills[0][s] += self.gamma * err * self.ppars[1][p] 
        self.ppars[0][p]  += self.gamma * err
        self.ppars[1][p]  += self.gamma * err * orig_skill
